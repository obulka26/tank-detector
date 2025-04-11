# Tank Detector

This project is a Machine Learning pipeline to detect tanks in images.

## Structure

- `model/`: training pipeline and saved model
- `api/`: FastAPI backend for predictions
- `frontend/`: simple interface (HTML/JS or Streamlit)
- `docs/`: documentation

## Goal

Train a model that can detect the presence of a tank in an image and provide an API and simple frontend for testing.

## Image Sources and Licenses

This project uses public image datasets for training:

1. **Cars and Tanks Image Classification Dataset**  
   - Author: Gateway Adam  
   - Source: [Kaggle](https://www.kaggle.com/datasets/gatewayadam/cars-and-tanks-image-classification)  
   - License: [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/)  
   - ✅ Free to use, modify, and distribute without attribution.

2. **Vehicle Detection Image Set**  
   - Author: Baris Dincer  
   - Source: [Kaggle](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set)  
   - License: [Open Data Commons Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)  
   - ✅ Free to use, including commercially, with attribution and compliance with the [ODbL license](https://opendatacommons.org/licenses/odbl/1-0/).

3. **HIIT Dataset**
   - Author: public haris  
   - Source: [Roboflow Universe](https://universe.roboflow.com/public-haris/hiit)  
   - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
   - ✅ Free to use, remix, and redistribute with proper attribution.  
   - BibTeX:
     ```bibtex
     @misc{hiit_dataset,
       title        = { HIIT Dataset },
       type         = { Open Source Dataset },
       author       = { public haris },
       howpublished = {\url{https://universe.roboflow.com/public-haris/hiit}},
       journal      = { Roboflow Universe },
       publisher    = { Roboflow },
       year         = { 2022 },
       month        = { jul },
       note         = { visited on 2025-04-10 }
     }
     ```

4. **Military Tank Detection Dataset**
   - Author: Keshav Memorial Institute of Technology  
   - Source: [Roboflow Universe](https://universe.roboflow.com/keshav-memorial-institute-of-technology-lbgah/military-tank-detection-rwlhy)  
   - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
   - ✅ Free to use, remix, and redistribute with proper attribution.  
   - BibTeX:
     ```bibtex
     @misc{military-tank-detection-rwlhy_dataset,
       title        = { military tank detection Dataset },
       type         = { Open Source Dataset },
       author       = { Keshav Memorial Institute of Technology },
       howpublished = {\url{https://universe.roboflow.com/keshav-memorial-institute-of-technology-lbgah/military-tank-detection-rwlhy}},
       journal      = { Roboflow Universe },
       publisher    = { Roboflow },
       year         = { 2022 },
       month        = { dec },
       note         = { visited on 2025-04-10 }
     }
     ```

## Dependencies and Licenses

This project uses the following libraries:

- **[PyTorch](https://pytorch.org/)** – BSD-style license  
- **[FastAPI](https://fastapi.tiangolo.com/)** – MIT license  
- **[Pillow](https://pillow.readthedocs.io/en/stable/)** – MIT-CMU license  
- **[Torchvision](https://pytorch.org/vision/stable/index.html)** – BSD 3-Clause license  
- **[Uvicorn](https://www.uvicorn.org/)** – BSD-3-Clause license
- **[Ultralytics YOLO](https://www.ultralytics.com/)** – [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) for non-commercial use. For commercial use, an [Ultralytics Enterprise License](https://www.ultralytics.com/) is required.
- **[OpenCV](https://opencv.org/)** – Apache 2.0 license  
- **[imageio](https://imageio.readthedocs.io/en/stable/)** – BSD 2-Clause license

## Status

🚧 In development

