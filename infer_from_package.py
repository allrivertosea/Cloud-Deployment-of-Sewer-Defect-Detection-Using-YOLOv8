from sewer_dt.infer import SewerDetector
detector = SewerDetector()
result = detector.detect(r'.\00005518.png')
print(result)