# Run Docker

```
docker build -t hooligan .
docker run --rm -v $(pwd):/usr/src/app -it hooligan 
```