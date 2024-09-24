## Natural Language Processing (NLP)
**MMAI5400-A1: web scraping reviews**
- Scraped 500+ reviews from Trustpilot.
- Written reviews to a CSV file with columns: companyName, datePublished, ratingValue, reviewBody.

**MMAI5400-A2: sentiment prediction for restaurants**
- Aimed to predict sentiment towards each restaurant based on customer reviews. 
- Utilize Python for sentiment segmentation and data cleaning.
- Applied TF-IDF to assess the importance of a word to one of the documents in a set or corpus. 
- Trained and tuned Support Vector Machines (SVMs) and Grid Search to ensure the accuracy of sentiment prediction.

**MMAI5400-A3: sentiment prediction for dishes**
- Aimed to predict sentiment towards dishes based on customer reviews. 
- Named Entity Recognition (NER) is used to locate the name of the meal.
- Trained and tuned Aspect Oriented Sentiment Analysis (ABSA) to predict the customer's sentiment and preference for the meal, theryby facilitating the identification of areas for improvement.


## Deep Learning
**A1: Neural Network Traning**

**A2: Object Detection**
- Data Preparation: Begins by unzipping and loading image data with annotations in COCO format (finished on Label Studio). The data is then converted into a format readable by Hugging Face's DatasetDict for object detection.
- Dataset Organization: The images and metadata (annotations) are organized into the correct folder structure for object detection. Unique category IDs are collected from the annotations to create id2label and label2id mappings.
- Model Preparation: The DETR (Detection Transformer) model is initialized from a pre-trained checkpoint (facebook/detr-resnet-50). An AutoImageProcessor is used to preprocess the images for the model.
- Training: The model is trained on the prepared dataset with specific training arguments such as batch size, number of epochs, learning rate, etc. The training progress is tracked and logged.
- Evaluation: After training, the model is evaluated on a test dataset to measure its performance using metrics like Average Precision (AP) and Average Recall (AR) at different IoU thresholds.
- Inference & Visualization: The trained model is used to make predictions on new images, detecting objects and displaying bounding boxes and labels. A function is also provided to count specific objects in an image.

**A3: Anomaly Detection**
- Data Preparation: Extracts frames from a video file and stores them as JPEG images. Loads and preprocesses the images for training the anomaly detection model.
- Model Training: Utilizes a Convolutional Autoencoder architecture for learning normal patterns in the image frames. Trains the autoencoder on the normal image frames to reconstruct them with minimal loss.
- Anomaly Detection: For each test frame, calculates the reconstruction loss compared to the original frame. Identifies frames with high reconstruction loss as anomalous, indicating they differ significantly from the learned normal patterns.
- Evaluation: Evaluates the model's performance by detecting anomalous frames in a given video or image set. Outputs the file names of anomalous frames for further analysis.
- Framework Used: Leverages Keras, a high-level neural networks API built on top of TensorFlow, for building and training the autoencoder model.
