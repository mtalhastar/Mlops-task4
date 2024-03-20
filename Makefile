build:
    docker build -t iris_classifier .

run:
    docker run -p 5000:5000 iris_classifier

stop:
    docker stop $(docker ps -q --filter ancestor=iris_classifier)

clean:
    docker system prune -f
