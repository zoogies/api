FROM python:3.9.0
WORKDIR /src/backend

COPY requirements.txt requirements.txt
RUN pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip3 install -r requirements.txt

COPY . .

# RUN echo "REMINDER >> Create src/backend/secret.txt && src/backend/backups/current.json" TODO get this to work

CMD ["gunicorn"  , "-b", "0.0.0.0:5055", "wgsi:app"]