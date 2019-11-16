# Donut - CPU

- OK

![](classification/images/donut1.png)

```
python3 classify_image_non_edge.py --model models/donut.tflite --label models/donut_labels.txt --input images/donut1.png
```

* Not OK *

![](classification/images/donut4.png)

```
python3 classify_image_non_edge.py --model models/donut.tflite --label models/donut_labels.txt --input images/donut4.png
```

# Donut - Edge

![](classification/images/donut2.png)

```
python3 classify_image.py --model models/donut_edge_tpu.tflite --label models/donut_labels.txt --input images/donut2.png
```

![](classification/images/donut6.png)

```
python3 classify_image.py --model models/donut_edge_tpu.tflite --label models/donut_labels.txt --input images/donut6.png
```