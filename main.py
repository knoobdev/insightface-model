import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import logging
from confluent_kafka import Consumer, TopicPartition, KafkaException
import json
import threading
import time

assert insightface.__version__>='0.7'

logger = logging.getLogger(__name__)

MODEL_NAME_DEFAULT = 'inswapper_128.onnx'
KAFKA_CONFIG = {
	'HOST': os.environ.get('KAFKA_HOST', '192.168.1.209:9093'),
	'TOPIC': os.environ.get('KAFKA_TOPIC', 'test-topic'),
	'GROUP_ID': os.environ.get('KAFKA_GROUP_ID', 'ahv-model'),
	'CONSUME_MESSAGE_SIZE': 3
}

app = FaceAnalysis()
app.prepare(ctx_id=0)
swapper = insightface.model_zoo.get_model(MODEL_NAME_DEFAULT, download=False, download_zip=False)

def handle_message(message):
	logger.error(f'Received message: {message.value().decode("utf-8")} from topic {message.topic()} partition {message.partition()}')
	start_time = time.time()
	img = ins_get_image('t1')
	faces = app.get(img)
	faces = sorted(faces, key = lambda x : x.bbox[0])
	source_face = faces[0]
	res = img.copy()
	for face in faces:
			res = swapper.get(res, face, source_face, paste_back=True)
	cv2.imwrite(f"./t{start_time}_swapped.jpg", res)
	end_time = time.time()
	elapsed_time = end_time - start_time
	logger.error(f"Execution time: {elapsed_time} seconds")
	time.sleep(150 / 1000)

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
						# Check for current assignments if no message is received
						current_assignments = consumer.assignment()
						if current_assignments:
								print(f"Currently assigned partitions: {current_assignments}")
						else:
								logger.error("No partitions currently assigned.")
						continue
				if message.error():
						if message.error().code() == KafkaException.PARTITION_EOF:
								# End of partition event
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
		# Close down consumer to commit final offsets.
		consumer.close()
