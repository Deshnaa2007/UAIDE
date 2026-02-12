import io
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

from detector import sliding_patch_scores, reconstruct_heatmap, rgb_to_gray, extract_residual, fft_stats, lbp_entropy
from ethical_assessment import EthicalAssessment, format_ethical_report, get_simple_status


def make_overlay_pil(img_arr, heatmap, alpha=0.5, cmap='jet'):
    # img_arr: HxWx3 in [0,1]
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(np.clip(img_arr, 0, 1))
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# Load the trained model (try different model files)
MODEL_PATH = None
MODEL = None
MODEL_INFO = None

# Look for saved model info files (improved and standard) and load the first valid one.
# Handles filenames produced by training code (e.g. '*_improved_info.pkl' or '*_info.pkl').
info_candidates = list(Path('.').glob('*_improved_info.pkl')) + list(Path('.').glob('*_info.pkl'))
info_candidates = sorted(info_candidates, key=lambda p: p.name, reverse=True)

for info_path in info_candidates:
    try:
        MODEL_INFO = joblib.load(str(info_path))
    except Exception:
        continue

    # If info is a dict with expected keys, attempt to load model
    if not isinstance(MODEL_INFO, dict) or 'model_type' not in MODEL_INFO:
        continue

    mtype = MODEL_INFO['model_type']
    try:
        if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved']:
            # instantiate the correct PyTorch model class
            if mtype == 'resnet':
                from train import DeepfakeResNet as _ModelClass
            elif mtype in ['fusion', 'fusion_improved']:
                from train import DeepfakeFeatureFusion as _ModelClass
            else:
                from train import DeepfakeCNN as _ModelClass

            MODEL = _ModelClass()

            # state_dict_path in info may be absolute or relative
            state_path = MODEL_INFO.get('state_dict_path')
            if state_path is None:
                # try common suffixes
                base = str(info_path).replace('_improved_info.pkl', '').replace('_info.pkl', '')
                candidates = [base + '_best_improved', base + '_best', base]
                for c in candidates:
                    if Path(c).exists():
                        state_path = c
                        break

            if state_path is None:
                raise FileNotFoundError('state_dict_path not found in model info and no candidate file exists')

            MODEL.load_state_dict(torch.load(state_path, map_location='cpu'))
            MODEL.eval()
            if torch.cuda.is_available():
                MODEL.to(torch.device('cuda'))

        else:
            # fallback: traditional sklearn/joblib model saved next to info
            base = str(info_path).replace('_improved_info.pkl', '').replace('_info.pkl', '')
            joblib_path = base + '.joblib'
            if Path(joblib_path).exists():
                MODEL = joblib.load(joblib_path)
            else:
                # try loading info directly as the model object
                MODEL = MODEL_INFO

        MODEL_PATH = str(info_path)
        print(f"Loaded model info: {MODEL_PATH} (type: {mtype})")
        break
    except Exception as e:
        print(f"Failed to instantiate model from {info_path}: {e}")
        MODEL = None
        MODEL_INFO = None
        continue


def extract_image_features_from_array(img_arr, patch_size=128, n_patches=8, random_state=None):
    # sample random patches (similar to training script) and pool mean/std
    H, W, _ = img_arr.shape
    patches = []
    rng = np.random.RandomState(random_state)
    for _ in range(n_patches):
        if H <= patch_size or W <= patch_size:
            y0 = max(0, (H - patch_size) // 2)
            x0 = max(0, (W - patch_size) // 2)
        else:
            y0 = int(rng.randint(0, H - patch_size + 1))
            x0 = int(rng.randint(0, W - patch_size + 1))
        patch = img_arr[y0:y0 + patch_size, x0:x0 + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            ph = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            ph[:patch.shape[0], :patch.shape[1]] = patch
            patch = ph
        patches.append(patch)

    feats = []
    for p in patches:
        g = rgb_to_gray(p)
        res = extract_residual(g)
        res_std = float(np.std(res))
        _, hf = fft_stats(g)
        # LBP expects integer images; convert to uint8 for stability
        ent = lbp_entropy((g * 255).astype(np.uint8))
        feats.append([res_std, hf, ent])
    feats = np.array(feats)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return np.concatenate([mean, std])[None, :]
    
def evaluate_model_on_validation(model, dataset_root='DeepfakeVsReal/Dataset'):
    p = Path(dataset_root)
    val_root = p / 'Validation'
    if not val_root.exists():
        return 'Validation folder not found under ' + str(p)

    real_folder = val_root / 'Real'
    fake_folder = val_root / 'Fake'
    files_real = sorted([str(x) for x in real_folder.rglob('*.jpg')] + [str(x) for x in real_folder.rglob('*.png')])
    files_fake = sorted([str(x) for x in fake_folder.rglob('*.jpg')] + [str(x) for x in fake_folder.rglob('*.png')])
    files = [(f, 0) for f in files_real] + [(f, 1) for f in files_fake]
    X = []
    y = []
    for f, lbl in files:
        try:
            pil = Image.open(f).convert('RGB')
            arr = np.asarray(pil).astype(np.float32) / 255.0
            feat = extract_image_features_from_array(arr, patch_size=128, n_patches=8, random_state=123)
            X.append(feat[0])
            y.append(lbl)
        except Exception as e:
            continue
    if len(X) == 0:
        return 'No validation images found or feature extraction failed.'
    X = np.array(X)
    y = np.array(y)
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X).astype(float)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = None
        report = classification_report(y, preds)
        lines = [f'Val accuracy: {acc:.4f}']
        if auc is not None:
            lines.append(f'Val ROC AUC: {auc:.4f}')
        lines.append('\n'+report)
        return '\n'.join(lines)
    except Exception as e:
        return f'Evaluation failed: {e}'


def predict_gradio(pil_img, ethical_threshold=0.5, show_raw_features=False):
    # pil_img is a PIL.Image
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    
    # Initialize ethical status
    ethical_status = "N/A"
    ethical_report = ""

    if MODEL is not None and MODEL_INFO is not None:
        try:
            if MODEL_INFO['model_type'] in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved']:
                # Deep learning model prediction with Grad-CAM
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Prepare image transform
                if MODEL_INFO['model_type'] in ['resnet', 'fusion', 'fusion_improved']:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                # Get prediction
                with torch.no_grad():
                    outputs = MODEL(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    prob_fake = float(probs[0, 1])
                    
                    # Use model's optimal threshold (0.371) for best balance
                    threshold = MODEL_INFO.get('optimal_threshold', 0.371)
                    pred_class = 1 if prob_fake >= threshold else 0

                label = 'AI-generated' if pred_class == 1 else 'Real (camera)'
                is_ai = pred_class == 1

                # Generate Grad-CAM visualization
                try:
                    from train import apply_gradcam_overlay
                    overlay_img, cam = apply_gradcam_overlay_from_pil(pil_img, MODEL, MODEL_INFO['model_type'])
                    overlay_pil = Image.fromarray(overlay_img)
                except Exception as e:
                    print(f"Grad-CAM failed: {e}")
                    # Fallback to traditional heatmap
                    patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
                    heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
                    overlay_pil = make_overlay_pil(img, heat)

                # Perform ethical assessment if AI-generated detected
                if is_ai:
                    assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
                    ethical_status = get_simple_status(assessment)
                    ethical_report = format_ethical_report(assessment)
                    if not show_raw_features:
                        idx = ethical_report.find('\nRAW FEATURES:')
                        if idx != -1:
                            ethical_report = ethical_report[:idx]

                return label, prob_fake, overlay_pil, ethical_status, ethical_report

            else:
                # Traditional ML model
                X = extract_image_features_from_array(img, patch_size=128, n_patches=8, random_state=0)
                if hasattr(MODEL, 'predict_proba'):
                    prob = float(MODEL.predict_proba(X)[:, 1][0])
                else:
                    pred = MODEL.predict(X)[0]
                    prob = float(pred)
                
                # Use model's optimal threshold for balanced detection
                threshold = MODEL_INFO.get('optimal_threshold', 0.371)
                is_ai = prob >= threshold
                label = 'AI-generated' if is_ai else 'Real (camera)'

                # Traditional heatmap
                patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
                heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
                overlay = make_overlay_pil(img, heat)

                # Perform ethical assessment if AI-generated detected
                if is_ai:
                    assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
                    ethical_status = get_simple_status(assessment)
                    ethical_report = format_ethical_report(assessment)
                    if not show_raw_features:
                        idx = ethical_report.find('\nRAW FEATURES:')
                        if idx != -1:
                            ethical_report = ethical_report[:idx]

                return label, prob, overlay, ethical_status, ethical_report

        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Fall back to heuristic

    # Fallback heuristic
    patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
    heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
    overlay = make_overlay_pil(img, heat)

    ai_score = float(np.mean(heat))
    is_ai = ai_score >= 0.5
    label = 'AI-generated' if is_ai else 'Real (camera)'
    
    # Perform ethical assessment if AI-generated detected
    if is_ai:
        assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
        ethical_status = get_simple_status(assessment)
        ethical_report = format_ethical_report(assessment)
        if not show_raw_features:
            idx = ethical_report.find('\nRAW FEATURES:')
            if idx != -1:
                ethical_report = ethical_report[:idx]
    
    return label, ai_score, overlay, ethical_status, ethical_report


def apply_gradcam_overlay_from_pil(pil_img, model, model_type):
    """Apply Grad-CAM to PIL image for deep learning models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare transform
    if model_type in ['resnet', 'fusion', 'fusion_improved']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_size = (224, 224)
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_size = (128, 128)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Get Grad-CAM
    from train import GradCAM
    if model_type == 'resnet':
        target_layer = model.resnet.layer4[-1].conv3
    elif model_type in ['fusion', 'fusion_improved']:
        # For fusion model, use ResNet's last conv layer
        target_layer = model.resnet[7][-1].conv3  # layer4 of ResNet
    else:
        target_layer = model.conv4  # Last conv layer of custom CNN

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor, target_class=1)  # Focus on fake class

    # Create overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize original image
    original = cv2.resize(np.array(pil_img), target_size)

    # Overlay
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay


title = "Advanced Deepfake Detection System with Ethical Assessment"

# Create balanced layout using Blocks
with gr.Blocks(title=title) as iface:
    gr.Markdown(f"""
    # {title}
    
    Upload an image to detect if it's AI-generated and assess its ethical status.
    
    **Current Model:** {MODEL_INFO['model_type'].upper() if MODEL_INFO else 'Heuristic-based'}
    """)
    
    with gr.Row():
        # Left Column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(type='pil', label='Upload Image', height=400)
            
            gr.Markdown("### Settings")
            ethical_threshold = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                step=0.01, 
                value=0.5, 
                label='Ethical Risk Threshold',
                info='Lower = more strict classification'
            )
            show_raw_features = gr.Checkbox(
                label='Show raw feature values', 
                value=False
            )
            
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            **Features:**
            - Deep Learning: CNN/ResNet with transfer learning
            - Grad-CAM visualization highlights suspicious regions
            - Ethical assessment evaluates privacy and misuse risks
            - Real-time GPU-accelerated inference
            """)
        
        # Right Column - Outputs
        with gr.Column(scale=1):
            gr.Markdown("### Detection Results")
            
            with gr.Row():
                detection_result = gr.Label(num_top_classes=2, label='Classification')
                ai_score = gr.Number(label='AI-likelihood Score', precision=4)
            
            heatmap = gr.Image(label='Detection Heatmap', height=400)
            
            gr.Markdown("### Ethical Assessment")
            ethical_status = gr.Textbox(label='Status', lines=2)
            
            with gr.Accordion("Full Report", open=False):
                ethical_report = gr.Textbox(
                    label='Detailed Assessment', 
                    lines=30
                )
    
    gr.Markdown("""
    ---
    **How it works:** The heatmap overlay shows regions the model considers suspicious for deepfake artifacts.
    Ethical classification is based on artifact detectability and human face presence.
    
    *Powered by FHIBE Dataset concepts for face authenticity verification.*
    """)
    
    # Connect button to function
    analyze_btn.click(
        fn=predict_gradio,
        inputs=[input_image, ethical_threshold, show_raw_features],
        outputs=[detection_result, ai_score, heatmap, ethical_status, ethical_report]
    )


if __name__ == '__main__':
    iface.launch()