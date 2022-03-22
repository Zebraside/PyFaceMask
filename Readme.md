# PyFaceMask

This project is an attempt to make realtime application to mask face (even on cpu).  
PyFaceMask is a playground to make poc of the application.

## Used models

### Face detection

[Face recognition](https://pypi.org/project/face-recognition/) is used to detect faces.  
Face detection can be used to reduce image size on cpu inference.

### Face segmentation

[Face segmentation](https://github.com/zllrunning/face-parsing.PyTorch)  based on BiSeNet v2.

### Style transfer

[Style transfer model](https://github.com/zfergus/face-preserving-style-transfer) with preserved face details.
