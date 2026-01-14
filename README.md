# Junk Food Ads Classifiers

A research project that evaluates multiple strategies for classifying junk food advertisements in images. This repository compares different vision models and classification approaches to classify unhealthy food marketing content.

## Classification Types

The project explores two classification paradigms:

- **Binary Classification**: Determines whether an image contains junk food advertising (junk food ad/non junk food ad)
- **Multi-label Classification**: Identifies specific categories of junk food present in advertisements (e.g., pizza, hamburger, soda...)

## Strategies

### Binary Classification

| Strategy      | Model                  | Framework   | Pretrained Weights  |
| ------------- | ---------------------- | ----------- | ------------------- |
| CLIP Binary   | ViT-B-16               | OpenCLIP    | `laion2b_s34b_b88k` |
| KNN Binary    | Multiple (see note ^1) | torchvision | ImageNet            |
| YOLO11 Binary | YOLO11m-cls            | Ultralytics | ImageNet            |

### Multi-label Classification

| Strategy         | Model                  | Framework   | Pretrained Weights  |
| ---------------- | ---------------------- | ----------- | ------------------- |
| CLIP Multi-label | ViT-B-16               | OpenCLIP    | `laion2b_s34b_b88k` |
| KNN Multi-label  | Multiple (see note ^1) | torchvision | ImageNet            |
| CNN Multi-label  | EfficientNetV2B0       | TensorFlow  | ImageNet            |

> **^1 KNN Feature Extractors**: The KNN notebooks evaluate multiple pretrained models as feature extractors: ResNeXt-101, EfficientNet V2, ConvNeXt, ViT, Swin Transformer, and DINOv2.

### Architecture Details

#### CLIP

Zero-shot classification by comparing image embeddings against text descriptions of junk food categories.

```mermaid
flowchart LR
    subgraph Input
        A[Image]
        B["Text Prompts<br/>templates per class<br/>(e.g. 'junk food ad', 'not a junk food ad')"]
    end

    subgraph CLIP["CLIP ViT-B-16"]
        C[Image Encoder<br/>Vision Transformer]
        D[Text Encoder<br/>Transformer]
    end

    subgraph Embeddings
        E[Image<br/>Embedding]
        F[Text<br/>Embeddings]
    end

    A --> C --> E
    B --> D --> F
    E --> G[Cosine<br/>Similarity]
    F --> G
    G --> H[Prediction]
```

#### k-NN

Extracts feature embeddings using pretrained models (see note ^1) and classifies based on similarity to labeled training examples.

```mermaid
flowchart LR
    subgraph Training
        A1[Training<br/>Images] --> B1[Pretrained<br/>Feature Extractor]
        B1 --> C1[Feature<br/>Store]
        L1[Labels] --> C1
    end

    subgraph Inference
        A2[Test<br/>Image] --> B2[Pretrained<br/>Feature Extractor]
        B2 --> D[Query<br/>Embedding]
    end

    C1 --> E[K-Nearest<br/>Neighbors]
    D --> E
    E --> F[Majority<br/>Vote]
    F --> G[Prediction]
```

#### CNN

Fine-tunes EfficientNetV2 with transfer learning for multi-label prediction.

```mermaid
flowchart LR
    A[Image] --> B["EfficientNetV2B0<br/>Backbone<br/>pooling='avg'"]
    B --> C[Dropout]
    C --> D[Dense Layer<br/>Sigmoid]
    D --> E[Multi-label<br/>Predictions]

```

#### YOLO 11

Leverages the classification variant of YOLO11 for efficient binary classification.

```mermaid
flowchart LR
    A[Image] --> B[YOLO11m-cls<br/>Backbone]
    B --> C[Feature<br/>Extraction]
    C --> D[Classification<br/>Head]
    D --> E[Softmax]
    E --> F[Binary<br/>Prediction]

```

## Getting Started

All notebooks are designed to run in Google Colab with GPU acceleration. Before running:

1. Enable GPU runtime in Colab (Runtime > Change runtime type > GPU)
2. Set up required secrets in Colab:
   - `ROBOFLOW_API_KEY`
   - `ROBOFLOW_WORKSPACE_ID`
   - `ROBOFLOW_PROJECT_ID`
   - `ROBOFLOW_DATASET_VERSION`

Due to copyright reasons, we cannot provide the dataset we used.
