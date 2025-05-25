import asyncio
import logging
import json
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

toxigen_hatebert = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-uncased")

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
INPUT_TOPIC = "input_hate_speech"
OUTPUT_TOPIC = "output_hate_speech"

async def consume_and_classify():
    consumer = AIOKafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="hate_detector_group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda m: json.dumps(m).encode("utf-8")
    )

    await consumer.start()
    await producer.start()
    try:
        async for msg in consumer:
            logging.info(f"HATE-SPEECH: Received message: {msg}")
            message = msg.value
            msg_id = message["id"]
            group_id = message["group_id"]
            content = message["content"]

            try:
                prediction = toxigen_hatebert(content)[0]
                logging.info(f"HATE-SPEECH: Got prediction: {prediction}")

                label = prediction.get("label", "")
                is_hate = label == "LABEL_1"
                if is_hate:
                    logging.info("HATE-SPEECH: Detected hate speech")
                else:
                    logging.info("HATE-SPEECH: Text is neutral")

                response = {
                    "id": msg_id, 
                    "group_id": group_id,
                    "is_hate_speech": is_hate,
                }
                await producer.send_and_wait(OUTPUT_TOPIC, value=response)
                logging.info(f"HATE-SPEECH: Processed message - {response}")

            except Exception as e:
                logging.error(f"HATE-SPEECH: Error processing message: {e}")
    finally:
        await consumer.stop()
        await producer.stop()

if __name__ == "__main__":
    logging.info("hate-speech-service starting")
    asyncio.run(consume_and_classify())
