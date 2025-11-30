IMAGE_NAME=sentiment-app
PORT=8501

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p $(PORT):8501 $(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)
