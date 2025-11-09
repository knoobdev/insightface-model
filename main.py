import os
import cv2
import insightface
import threading
import time
import requests
import logging
import json
import os.path as osp
from pathlib import Path
from insightface.app import FaceAnalysis
from confluent_kafka import Consumer, KafkaException
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Tuple, Optional, Callable

assert insightface.__version__>='0.7'

logger = logging.getLogger(__name__)

MODEL_NAME_DEFAULT = 'inswapper_128.onnx'
KAFKA_CONFIG = {
	'HOST': os.environ.get('KAFKA_HOST', '192.168.1.209:9093'),
	'TOPIC': os.environ.get('KAFKA_TOPIC', 'test-topic'),
	'GROUP_ID': os.environ.get('KAFKA_GROUP_ID', 'ahv-model'),
	'CONSUME_MESSAGE_SIZE': os.environ.get('KAFKA_CONSUME_MESSAGE_SIZE', 3),
}

app = FaceAnalysis()
app.prepare(ctx_id=0)
swapper = insightface.model_zoo.get_model(MODEL_NAME_DEFAULT, download=False, download_zip=False)

class ImageCache:
    data = {}

def get_image(image_file, to_rgb=False, use_cache=True):
	key = (image_file, to_rgb)
	if key in ImageCache.data:
			return ImageCache.data[key]
	if osp.exists(image_file):
		img = cv2.imread(image_file)
		if to_rgb:
				img = img[:,:,::-1]
		if use_cache:
				ImageCache.data[key] = img
		return img
	assert image_file is not None, '%s not found'%image_file

def download_files(
    urls: Dict[str, str],
    output_dir: str = ".",
    max_workers_per_file: int = 4,
    max_concurrent_files: int = 3,
    chunk_size: int = 2 * 1024 * 1024,
    max_retries: int = 5,
    timeout: int = 60,
    verify_ssl: bool = False,
    on_complete: Optional[Callable] = None,
    on_error: Optional[Callable] = None
) -> Dict[str, Optional[str]]:
    """    
    Args:
        urls: Dict {key: url} - key để identify file, url để download
        output_dir: Thư mục lưu file
        max_workers_per_file: Số luồng song song cho mỗi file
        max_concurrent_files: Số file download đồng thời
        chunk_size: Kích thước chunk (bytes)
        max_retries: Số lần retry
        timeout: Timeout (giây)
        verify_ssl: Xác thực SSL
        on_complete: Callback khi 1 file hoàn tất: fn(key, path, size, duration)
        on_error: Callback khi 1 file lỗi: fn(key, error)
        
    Example:
        >>> results = download_files({
        ...     'sourceURL': 'https://example.com/source.zip',
        ...     'targetURL': 'https://example.com/target.zip',
        ... })
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    results = {}
    lock = threading.Lock()
    completed_files = [0]
    
    def create_session():
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_workers_per_file,
            pool_maxsize=max_workers_per_file * 2
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        })
        return session
    
    def download_single_file(key: str, url: str) -> Tuple[str, Optional[str], Optional[Exception]]:
        filename = Path(url).name or f"{key}_file"
        output_path = output_dir / filename
        temp_dir = output_dir / f".{filename}.parts"
        file_start_time = time.time()
        
        try:
            # Lấy file size
            session = create_session()
            try:
                resp = session.head(url, timeout=timeout, verify=verify_ssl, allow_redirects=True)
            except:
                resp = session.get(url, stream=True, timeout=timeout, verify=verify_ssl)
                resp.close()
            
            total_size = int(resp.headers.get('Content-Length', 0))
            support_range = resp.headers.get('Accept-Ranges', 'none') != 'none'
            
            if not support_range or total_size < chunk_size:
                resp = session.get(url, stream=True, timeout=timeout, verify=verify_ssl)
                with open(output_path, 'wb') as f:
                    for data in resp.iter_content(chunk_size=8192):
                        if data:
                            f.write(data)
            else:
                temp_dir.mkdir(exist_ok=True)
                downloaded = [0]
                
                def download_chunk(start, end, chunk_id):
                    chunk_file = temp_dir / f"chunk_{chunk_id}"
                    
                    if chunk_file.exists() and chunk_file.stat().st_size == (end - start + 1):
                        return True
                    
                    headers = {'Range': f'bytes={start}-{end}'}
                    chunk_session = create_session()
                    
                    for attempt in range(max_retries):
                        try:
                            r = chunk_session.get(url, headers=headers, stream=True, 
                                                 timeout=timeout, verify=verify_ssl)
                            r.raise_for_status()
                            
                            with open(chunk_file, 'wb') as f:
                                for data in r.iter_content(4096):
                                    if data:
                                        f.write(data)
                                        downloaded[0] += len(data)
                            return True
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(min(2 ** attempt, 32))
                            if chunk_file.exists():
                                chunk_file.unlink()
                    return False
                num_chunks = min(max_workers_per_file, (total_size + chunk_size - 1) // chunk_size)
                chunk_size_adj = total_size // num_chunks
                ranges = [
                    (i * chunk_size_adj, 
                     min((i + 1) * chunk_size_adj - 1, total_size - 1) if i < num_chunks - 1 else total_size - 1,
                     i)
                    for i in range(num_chunks)
                ]
                
                with ThreadPoolExecutor(max_workers=max_workers_per_file) as executor:
                    futures = [executor.submit(download_chunk, s, e, i) for s, e, i in ranges]
                    for future in as_completed(futures):
                        if not future.result():
                            raise Exception("Chunk download failed")
                with open(output_path, 'wb') as out:
                    for i in range(num_chunks):
                        chunk_file = temp_dir / f"chunk_{i}"
                        with open(chunk_file, 'rb') as chunk:
                            out.write(chunk.read())
                        chunk_file.unlink()
                temp_dir.rmdir()
            
            duration = time.time() - file_start_time
            
            with lock:
                completed_files[0] += 1
            if on_complete:
                on_complete(key, str(output_path), total_size, duration)
            return key, str(output_path), None
        except Exception as e:
            with lock:
                completed_files[0] += 1
            if on_error:
                on_error(key, e)
            if output_path.exists():
                output_path.unlink()
            if temp_dir.exists():
                for f in temp_dir.glob("*"):
                    f.unlink()
                temp_dir.rmdir()
            
            return key, None, e
    with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
        futures = {
            executor.submit(download_single_file, key, url): key
            for key, url in urls.items()
        }
        
        for future in as_completed(futures):
            key, path, error = future.result()
            results[key] = path
    
    return results

def on_error(key, e):
  logger.error(f"{key}:{e}")

def handle_message(message):
	try:
		handle_data = json.loads(message.value().decode('utf-8'))
		model_params = handle_data['modelParams']
		results = download_files(
			urls={
					'source_url': model_params['sourceUrl'],
					'target_url': model_params['targetUrl']
			},
			output_dir="/root/.insightface/files",
			chunk_size=2 * 1024 * 1024,
			timeout=10,
      on_error=on_error
		)
		
		if results['source_url'] is None:
			logger.error(f"source_url not found!")
			return None
		if results['target_url'] is None:
			logger.error(f"target_url not found!")
			return None
		start_time = time.time()
		source_image = get_image(results['source_url'])
		faces = app.get(source_image)
		faces = sorted(faces, key = lambda x : x.bbox[0])
		source_face = faces[0]
		target_image = get_image(results['target_url'])
		result_image = target_image.copy()
		for face in faces:
				result_image = swapper.get(result_image, face, source_face, paste_back=True)
		cv2.imwrite(f"./t{start_time}_swapped.jpg", result_image)
		end_time = time.time()
		elapsed_time = end_time - start_time
		logger.error(f"Execution time: {elapsed_time} seconds")
		time.sleep(150 / 1000)
	except UnicodeDecodeError:
			logger.error(f"Error decoding message value as UTF-8: {message.value()}")
	except json.JSONDecodeError:
			logger.error(f"Error parsing message value as JSON: {message.value().decode('utf-8', errors='ignore')}")

if __name__ == '__main__':
	consumer = Consumer({
			'bootstrap.servers': KAFKA_CONFIG['HOST'],
			'group.id': KAFKA_CONFIG['GROUP_ID'],
			'auto.offset.reset': 'earliest',
			'compression.type': 'gzip'
	})

	consumer.subscribe([KAFKA_CONFIG['TOPIC']])
	try:
		while True:
			time.sleep(200 / 1000)
			messages = consumer.consume(KAFKA_CONFIG['CONSUME_MESSAGE_SIZE'], timeout=1)
			threads = []
			if messages is None:
				continue
			if not messages:
        # No messages received
				continue
			for message in messages:
				if message is None:
						current_assignments = consumer.assignment()
						if current_assignments:
								print(f"Currently assigned partitions: {current_assignments}")
						else:
								logger.error("No partitions currently assigned.")
						continue
				if message.error():
						if message.error().code() == KafkaException.PARTITION_EOF:
								logger.error(f'%% {message.topic()} [{message.partition()}] reached end offset {message.offset()}')
						else:
								logger.error(f'Error: {message.error()}')
						continue
				thread = threading.Thread(target=handle_message, args=(message,))
				threads.append(thread)
				thread.start()
			for thread in threads:
				thread.join()
	except KeyboardInterrupt:
		pass
	finally:
		consumer.close()
