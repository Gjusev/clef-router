# clef-router proxy container.
#
# Build:  docker build -t clef-router .
# Run:    docker run -p 8000:8000 -e CLEF_ACCOUNT_ID=... -e CLEF_API_TOKEN=... clef-router
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["clef-router", "--host", "0.0.0.0"]
