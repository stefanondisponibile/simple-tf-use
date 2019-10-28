# simple-tf-use
[Simple-TensorFlow-Serving](https://github.com/tobegit3hub/simple_tensorflow_serving) powered [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1) [Multilingual, Large]

---

## Install.
Create a virtual environment or install requirements to your virtual environment.
You could `pip install -r requirements.txt`, but dependencies basically are:

`tensorflow==1.13.1`
`tf-sentencepiece==tf-sentencepiece==0.1.83`

## Build.
`build.py` should handle everything:

```
python build.py
```

Re-build the container:
```
cd simple_tensorflow_serving && docker build . --tag stefanondisponibile/simple-tf-use:latest
```

Run the container: 
```
docker run -dp 8501:8501 stefanondisponibile/simple-tf-use:latest
```

Test it: 
```
curl -H "Content-Type: application/json" -X POST -d '{"model_name": "default", "model_version": "001", "data": { "text": ["Some text."] }}' http://localhost:8501
```

Looking good? Push it to DockerHUB: 
```
docker push stefanondisponibile/simple-tf-use
```
