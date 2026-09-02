import os
import subprocess
import tempfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["Normal", "Violence", "Weaponized"]
CLASS_AR = {
    "Normal": "طبيعي",
    "Violence": "عنف",
    "Weaponized": "عنف مسلح",
}
COLORS = {
    "Normal": "#22C55E",
    "Violence": "#EF4444",
    "Weaponized": "#F59E0B",
}
NUM_FRAMES = 16
IMG_SIZE = 112
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            resnet = models.resnet18(weights=None)
        except TypeError:
            resnet = models.resnet18(pretrained=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(
            512, 256, num_layers=2, batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        x = x.reshape(batch * time, channels, height, width)
        with torch.no_grad():
            features = self.cnn(x)
        features = features.reshape(batch, time, -1)
        lstm_out, _ = self.lstm(features)
        return self.classifier(lstm_out[:, -1, :])


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"لم يتم العثور على {MODEL_PATH}. ضعه بجانب ملف الواجهة."
        )
    loaded_model = CNNLSTMModel().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()
    return loaded_model


MODEL = load_model()
DETECTOR = None


def get_detector():
    """Load the detector only when annotated-video output is requested."""
    global DETECTOR
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "مكتبة ultralytics غير مثبتة. نفّذ: pip install ultralytics"
        )
    if DETECTOR is None:
        DETECTOR = YOLO("yolo11n.pt")
    return DETECTOR


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def read_window_frames(cap, fps, start_sec, end_sec):
    if end_sec <= start_sec:
        end_sec = start_sec + (1.0 / max(fps, 1.0))

    sample_times = np.linspace(
        start_sec,
        end_sec,
        NUM_FRAMES,
        endpoint=False,
        dtype=np.float32,
    )
    frames = []
    last_frame = None

    for timestamp in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = transform(frame)
            frames.append(last_frame)
        elif last_frame is not None:
            frames.append(last_frame.clone())
        else:
            frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

    return torch.stack(frames)


def predict_windows(video_path, window_seconds, stride_seconds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("تعذر فتح الفيديو.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("الفيديو لا يحتوي على معلومات زمنية صالحة.")

    duration = frame_count / fps
    starts = np.arange(0.0, max(duration, 0.001), stride_seconds)
    rows = []
    batch_size = 16 if DEVICE.type == "cuda" else 4
    with torch.inference_mode():
        for index in range(0, len(starts), batch_size):
            batch_starts = starts[index : index + batch_size]
            metadata = [
                (
                    float(start),
                    min(float(start + window_seconds), duration),
                )
                for start in batch_starts
            ]
            batch = torch.stack(
                [
                    read_window_frames(cap, fps, start, end)
                    for start, end in metadata
                ]
            ).to(DEVICE)
            probabilities = torch.softmax(MODEL(batch), dim=1)
            probabilities = probabilities.cpu().numpy()

            for (start, end), window_probabilities in zip(
                metadata, probabilities
            ):
                predicted_index = int(np.argmax(window_probabilities))
                predicted_class = CLASSES[predicted_index]
                rows.append(
                    {
                        "start": start,
                        "end": end,
                        "center": (start + end) / 2.0,
                        "predicted_class": predicted_class,
                        "confidence": float(
                            window_probabilities[predicted_index]
                        ),
                        **{
                            class_name: float(
                                window_probabilities[class_index]
                            )
                            for class_index, class_name in enumerate(CLASSES)
                        },
                    }
                )
    cap.release()
    return rows, duration


def smooth_probabilities(rows):
    if len(rows) < 3:
        return rows

    values = np.array(
        [[row[class_name] for class_name in CLASSES] for row in rows],
        dtype=np.float32,
    )
    padded = np.pad(values, ((1, 1), (0, 0)), mode="edge")
    smoothed = (
        padded[:-2] * 0.25 + padded[1:-1] * 0.50 + padded[2:] * 0.25
    )

    for row, probabilities in zip(rows, smoothed):
        predicted_index = int(np.argmax(probabilities))
        row.update(
            {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(CLASSES)
            }
        )
        row["predicted_class"] = CLASSES[predicted_index]
        row["confidence"] = float(probabilities[predicted_index])
    return rows


def find_events(rows, alert_threshold):
    events = []
    active_event = None

    for row in rows:
        threat_class = max(
            ("Violence", "Weaponized"),
            key=lambda class_name: row[class_name],
        )
        threat_probability = row[threat_class]
        is_threat = (
            row["predicted_class"] != "Normal"
            and threat_probability >= alert_threshold
        )

        if is_threat:
            if active_event is None:
                active_event = {
                    "start": row["start"],
                    "end": row["end"],
                    "classes": [threat_class],
                    "peak": threat_probability,
                }
            else:
                active_event["end"] = row["end"]
                active_event["classes"].append(threat_class)
                active_event["peak"] = max(
                    active_event["peak"], threat_probability
                )
        elif active_event is not None:
            events.append(active_event)
            active_event = None

    if active_event is not None:
        events.append(active_event)

    for event in events:
        event["class"] = max(
            ("Violence", "Weaponized"),
            key=event["classes"].count,
        )
    return events


def build_chart(rows, events):
    figure = go.Figure()
    for class_name in CLASSES:
        figure.add_trace(
            go.Scatter(
                x=[row["center"] for row in rows],
                y=[row[class_name] * 100 for row in rows],
                name=CLASS_AR[class_name],
                mode="lines+markers",
                line={"color": COLORS[class_name], "width": 3},
                marker={"size": 5},
                hovertemplate=(
                    "الوقت: %{x:.1f} ثانية<br>"
                    + CLASS_AR[class_name]
                    + ": %{y:.1f}%<extra></extra>"
                ),
            )
        )

    for event in events:
        figure.add_vrect(
            x0=event["start"],
            x1=event["end"],
            fillcolor=COLORS[event["class"]],
            opacity=0.12,
            line_width=0,
        )

    figure.update_layout(
        template="plotly_dark",
        autosize=True,
        height=430,
        margin={"l": 50, "r": 20, "t": 55, "b": 50},
        title="نسب التصنيف عبر زمن الفيديو",
        xaxis_title="الوقت (ثانية)",
        yaxis_title="الاحتمال (%)",
        yaxis={"range": [0, 100]},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        paper_bgcolor="#09111f",
        plot_bgcolor="#0f1b2d",
    )
    return figure


def build_timeline(events, duration):
    if not events:
        event_html = """
        <div class="safe-card">
            <div class="safe-icon">✓</div>
            <div><b>لم يتم اكتشاف أحداث عنف</b><br>
            <span>جميع المقاطع أقل من حد التنبيه المحدد.</span></div>
        </div>
        """
    else:
        cards = []
        for number, event in enumerate(events, start=1):
            css_class = (
                "weapon-event"
                if event["class"] == "Weaponized"
                else "violence-event"
            )
            cards.append(
                f"""
                <div class="event-card {css_class}">
                    <b>حدث {number}: {CLASS_AR[event["class"]]}</b>
                    <span>{format_time(event["start"])} ← {format_time(event["end"])}</span>
                    <small>أعلى ثقة: {event["peak"] * 100:.1f}%</small>
                </div>
                """
            )
        event_html = "".join(cards)

    markers = "".join(
        f"""
        <div class="timeline-event"
             style="left:{event["start"] / max(duration, 0.001) * 100:.2f}%;
                    width:{max((event["end"] - event["start"]) / max(duration, 0.001) * 100, 0.8):.2f}%;
                    background:{COLORS[event["class"]]};"
             title="{CLASS_AR[event["class"]]}: {format_time(event["start"])} - {format_time(event["end"])}">
        </div>
        """
        for event in events
    )

    timeline_html = f"""
    <div class="timeline-panel">
        <div class="timeline-heading">الخط الزمني للأحداث</div>
        <div class="timeline-track">{markers}</div>
        <div class="timeline-labels">
            <span>00:00.00</span><span>{format_time(duration)}</span>
        </div>
        <div class="event-list">{event_html}</div>
    </div>
    """
    return " ".join(
        line.strip() for line in timeline_html.splitlines() if line.strip()
    )


def build_summary(rows, events, duration):
    if events:
        first_event = events[0]
        status = "تم رصد تهديد"
        status_color = "#EF4444"
        first_text = format_time(first_event["start"])
    else:
        status = "الفيديو آمن"
        status_color = "#22C55E"
        first_text = "لا يوجد"

    overall_class = max(
        CLASSES,
        key=lambda class_name: np.mean(
            [row[class_name] for row in rows]
        ),
    )
    return f"""
    <div class="summary-grid">
        <div class="metric-card">
            <small>الحالة العامة</small>
            <b style="color:{status_color}">{status}</b>
        </div>
        <div class="metric-card">
            <small>التصنيف الغالب</small>
            <b>{CLASS_AR[overall_class]}</b>
        </div>
        <div class="metric-card">
            <small>بداية أول تهديد</small>
            <b>{first_text}</b>
        </div>
        <div class="metric-card">
            <small>مدة الفيديو</small>
            <b>{format_time(duration)}</b>
        </div>
    </div>
    """


def build_details(rows):
    records = []
    for index, row in enumerate(rows, start=1):
        records.append(
            {
                "المقطع": index,
                "من": format_time(row["start"]),
                "إلى": format_time(row["end"]),
                "النتيجة": CLASS_AR[row["predicted_class"]],
                "الثقة": f'{row["confidence"] * 100:.1f}%',
                "طبيعي": f'{row["Normal"] * 100:.1f}%',
                "عنف": f'{row["Violence"] * 100:.1f}%',
                "عنف مسلح": f'{row["Weaponized"] * 100:.1f}%',
            }
        )
    return pd.DataFrame(records)


def row_at_time(rows, timestamp):
    """Return the temporal prediction that best represents this frame."""
    containing = [
        row
        for row in rows
        if row["start"] <= timestamp < row["end"]
    ]
    if containing:
        return max(containing, key=lambda row: row["confidence"])
    return min(rows, key=lambda row: abs(row["center"] - timestamp))


def create_annotated_video(video_path, rows, alert_threshold, progress=None):
    detector = get_detector()
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("تعذر إنشاء الفيديو المعلّم.")

    output_dir = tempfile.mkdtemp(prefix="scvd_analysis_")
    raw_output_path = os.path.join(output_dir, "scvd_annotated_raw.mp4")
    output_path = os.path.join(output_dir, "scvd_annotated.mp4")
    writer = cv2.VideoWriter(
        raw_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("تعذر فتح VideoWriter لإنتاج الفيديو.")

    detect_every = max(1, int(round(fps / 6.0)))
    last_boxes = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_index / fps
        temporal_row = row_at_time(rows, timestamp)
        threat_class = max(
            ("Violence", "Weaponized"),
            key=lambda name: temporal_row[name],
        )
        threat_probability = temporal_row[threat_class]
        threat_active = (
            temporal_row["predicted_class"] != "Normal"
            and threat_probability >= alert_threshold
        )

        if threat_active and frame_index % detect_every == 0:
            result = detector.predict(
                source=frame,
                classes=[0],
                conf=0.30,
                imgsz=640,
                verbose=False,
                device=0 if DEVICE.type == "cuda" else "cpu",
            )[0]
            last_boxes = (
                result.boxes.xyxy.cpu().numpy().astype(int).tolist()
                if result.boxes is not None
                else []
            )
        elif not threat_active:
            last_boxes = []

        if threat_active:
            bgr_color = (
                (0, 165, 255)
                if threat_class == "Weaponized"
                else (40, 40, 235)
            )
            label = (
                f"{threat_class.upper()}  {threat_probability * 100:.1f}%"
            )

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 72), bgr_color, -1)
            frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
            cv2.putText(
                frame,
                label,
                (22, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"TIME {format_time(timestamp)}",
                (22, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            for x1, y1, x2, y2 in last_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 3)
                caption_y = max(24, y1 - 8)
                cv2.rectangle(
                    frame,
                    (x1, caption_y - 22),
                    (min(x2, x1 + 185), caption_y + 3),
                    bgr_color,
                    -1,
                )
                cv2.putText(
                    frame,
                    "PERSON IN ALERT SCENE",
                    (x1 + 4, caption_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.rectangle(frame, (14, 14), (174, 52), (28, 130, 65), -1)
            cv2.putText(
                frame,
                "NORMAL",
                (28, 41),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_index += 1
        if progress and frame_index % max(1, int(fps)) == 0:
            ratio = frame_index / max(frame_count, 1)
            progress(
                0.76 + min(ratio, 1.0) * 0.21,
                desc="إنشاء الفيديو المعلّم",
            )

    cap.release()
    writer.release()

    # Browsers do not reliably play OpenCV's mp4v output. Convert it to
    # H.264/yuv420p and move metadata to the beginning for web streaming.
    try:
        conversion = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_output_path,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if conversion.returncode == 0 and os.path.getsize(output_path) > 0:
            os.remove(raw_output_path)
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return raw_output_path


def build_explanation(rows, events, duration, alert_threshold):
    normal_seconds = 0.0
    violence_seconds = 0.0
    weaponized_seconds = 0.0
    for row in rows:
        segment_duration = max(0.0, row["end"] - row["start"])
        if row["predicted_class"] == "Normal":
            normal_seconds += segment_duration
        elif row["predicted_class"] == "Violence":
            violence_seconds += segment_duration
        else:
            weaponized_seconds += segment_duration

    # Overlapping windows make raw totals larger than the video, so normalize them.
    raw_total = normal_seconds + violence_seconds + weaponized_seconds
    scale = duration / raw_total if raw_total > 0 else 0.0
    normal_seconds *= scale
    violence_seconds *= scale
    weaponized_seconds *= scale

    highest_threat = max(
        (
            (row["Violence"], "Violence", row["center"])
            for row in rows
        ),
        default=(0.0, "Violence", 0.0),
    )
    highest_weapon = max(
        (
            (row["Weaponized"], "Weaponized", row["center"])
            for row in rows
        ),
        default=(0.0, "Weaponized", 0.0),
    )
    peak = max(highest_threat, highest_weapon, key=lambda item: item[0])

    if not events:
        risk_level = "منخفض"
        headline = "لم يكتشف النظام حدث عنف متماسك يتجاوز حد التنبيه."
    elif any(event["class"] == "Weaponized" for event in events):
        risk_level = "مرتفع"
        headline = "اكتشف النظام حدثًا واحدًا على الأقل باحتمال عنف مسلح."
    elif len(events) >= 2 or peak[0] >= 0.80:
        risk_level = "مرتفع"
        headline = "اكتشف النظام عدة إشارات عنف أو إشارة ذات ثقة مرتفعة."
    else:
        risk_level = "متوسط"
        headline = "اكتشف النظام مقطعًا يحتمل احتواءه على عنف."

    event_cards = []
    for number, event in enumerate(events, start=1):
        event_color = COLORS[event["class"]]
        event_cards.append(
            f"""
            <div class="report-event" style="border-right-color:{event_color}">
                <div class="report-event-number">الحدث {number}</div>
                <div class="report-event-class">{CLASS_AR[event["class"]]}</div>
                <div class="report-event-time">
                    {format_time(event["start"])} — {format_time(event["end"])}
                </div>
                <div class="report-event-confidence">
                    أعلى ثقة <b>{event["peak"] * 100:.1f}%</b>
                </div>
            </div>
            """
        )
    if not event_cards:
        event_cards.append(
            """
            <div class="report-safe">
                <span class="report-safe-icon">✓</span>
                <div>
                    <b>لا توجد أحداث مسجلة</b>
                    <small>لم تتجاوز المقاطع حد التنبيه المحدد.</small>
                </div>
            </div>
            """
        )

    risk_color = {
        "منخفض": "#22C55E",
        "متوسط": "#F59E0B",
        "مرتفع": "#EF4444",
    }[risk_level]
    explanation_html = f"""
    <section class="analysis-report">
        <div class="report-head">
            <div>
                <span class="report-eyebrow">SCVD • AUTOMATED VIDEO REPORT</span>
                <h2>تقرير تحليل الفيديو</h2>
                <p>{headline}</p>
            </div>
            <div class="risk-badge" style="--risk-color:{risk_color}">
                <small>مستوى الخطورة</small>
                <b>{risk_level}</b>
            </div>
        </div>

        <div class="report-kpis">
            <div><small>مدة الفيديو</small><b>{format_time(duration)}</b></div>
            <div><small>عدد الأحداث</small><b>{len(events)}</b></div>
            <div><small>أعلى إشارة</small><b>{peak[0] * 100:.1f}%</b></div>
            <div><small>وقت الذروة</small><b>{format_time(peak[2])}</b></div>
        </div>

        <div class="report-section">
            <div class="report-section-title">الملخص التنفيذي</div>
            <p>
                حلّل النظام الفيديو على هيئة مقاطع زمنية متداخلة. كانت أعلى
                إشارة مصنفة كـ <b>{CLASS_AR[peak[1]]}</b> بنسبة
                <b>{peak[0] * 100:.1f}%</b> تقريبًا عند
                <b>{format_time(peak[2])}</b>. استُخدم حد تنبيه
                <b>{alert_threshold * 100:.0f}%</b> لدمج المقاطع المتتابعة
                وتحويلها إلى أحداث قابلة للمراجعة.
            </p>
        </div>

        <div class="duration-grid">
            <div class="duration-item normal-duration">
                <span>طبيعي</span><b>{normal_seconds:.1f} ثانية</b>
            </div>
            <div class="duration-item violence-duration">
                <span>عنف</span><b>{violence_seconds:.1f} ثانية</b>
            </div>
            <div class="duration-item weapon-duration">
                <span>عنف مسلح</span><b>{weaponized_seconds:.1f} ثانية</b>
            </div>
        </div>

        <div class="report-section">
            <div class="report-section-title">الأحداث المكتشفة</div>
            <div class="report-events">{"".join(event_cards)}</div>
        </div>

        <div class="report-notice">
            <b>تنبيه تفسيري:</b>
            النتيجة أداة مساعدة للمراجعة وليست حكمًا نهائيًا. البوكس يحدد
            الأشخاص الموجودين داخل مشهد التنبيه، ولا يحدد المعتدي أو يثبت
            ارتكاب فعل عنيف.
        </div>
    </section>
    """
    # Keep the fragment on one logical line. Markdown treats indented HTML
    # following a blank line as a code block, which exposes the tags in
    # Streamlit instead of rendering the report.
    explanation_html = " ".join(
        line.strip() for line in explanation_html.splitlines() if line.strip()
    )
    return explanation_html, {
        "duration_seconds": duration,
        "risk_level": risk_level,
        "headline_ar": headline,
        "events_count": len(events),
        "normal_seconds_estimate": normal_seconds,
        "violence_seconds_estimate": violence_seconds,
        "weaponized_seconds_estimate": weaponized_seconds,
        "peak_threat": {
            "class": peak[1],
            "probability": peak[0],
            "time_seconds": peak[2],
        },
    }


def create_report_file(rows, events, summary_data, explanation):
    output_dir = tempfile.mkdtemp(prefix="scvd_report_")
    report_path = os.path.join(output_dir, "SCVD_analysis_report.html")
    event_rows = "".join(
        f"""
        <tr>
            <td>{index}</td>
            <td>{CLASS_AR[event["class"]]}</td>
            <td>{format_time(event["start"])}</td>
            <td>{format_time(event["end"])}</td>
            <td>{event["peak"] * 100:.1f}%</td>
        </tr>
        """
        for index, event in enumerate(events, start=1)
    )
    if not event_rows:
        event_rows = (
            '<tr><td colspan="5">لا توجد أحداث تجاوزت حد التنبيه.</td></tr>'
        )
    report_document = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SCVD Analysis Report</title>
<style>
body{{font-family:Arial,Tahoma,sans-serif;background:#eef2f7;color:#172033;margin:0;padding:32px}}
.page{{max-width:980px;margin:auto;background:#fff;border-radius:18px;padding:34px;box-shadow:0 12px 35px #1e293b20}}
h1{{margin:0 0 8px;color:#0f2d52}} .meta{{color:#64748b;margin-bottom:26px}}
.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}}
.card{{background:#f7f9fc;border:1px solid #dce4ef;border-radius:12px;padding:16px}}
.card small{{display:block;color:#64748b;margin-bottom:6px}} .card b{{font-size:20px}}
table{{width:100%;border-collapse:collapse;margin-top:14px}} th{{background:#0f2d52;color:#fff}}
th,td{{padding:12px;border:1px solid #dde4ee;text-align:right}} tr:nth-child(even){{background:#f8fafc}}
.notice{{margin-top:24px;padding:16px;background:#fff7ed;border-right:4px solid #f59e0b;border-radius:8px}}
</style>
</head>
<body><main class="page">
<h1>تقرير تحليل فيديو المراقبة</h1>
<div class="meta">SCVD CNN + LSTM • Temporal Analysis • Person Detection</div>
<p>{summary_data["headline_ar"]}</p>
<div class="summary">
<div class="card"><small>مدة الفيديو</small><b>{format_time(summary_data["duration_seconds"])}</b></div>
<div class="card"><small>مستوى الخطورة</small><b>{summary_data["risk_level"]}</b></div>
<div class="card"><small>عدد الأحداث</small><b>{summary_data["events_count"]}</b></div>
</div>
<h2>سجل الأحداث</h2>
<table><thead><tr><th>#</th><th>التصنيف</th><th>البداية</th><th>النهاية</th><th>أعلى ثقة</th></tr></thead>
<tbody>{event_rows}</tbody></table>
<div class="notice"><b>ملاحظة منهجية:</b> هذه النتائج آلية ومخصصة لدعم المراجعة البشرية.
البوكس يحدد الأشخاص الموجودين في مشهد التنبيه ولا يثبت أن الشخص هو المعتدي.</div>
</main></body></html>"""
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_document)
    return report_path


def analyze_video(
    video_path,
    window_seconds,
    stride_seconds,
    alert_threshold_percent,
    make_annotated_video,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("ارفع فيديو أولًا.")
    if stride_seconds > window_seconds:
        raise gr.Error("خطوة التحليل يجب ألا تكون أكبر من طول المقطع.")

    progress(0.05, desc="قراءة الفيديو")
    rows, duration = predict_windows(
        video_path,
        float(window_seconds),
        float(stride_seconds),
    )
    progress(0.75, desc="تحليل الخط الزمني")
    rows = smooth_probabilities(rows)
    events = find_events(rows, float(alert_threshold_percent) / 100.0)

    threshold = float(alert_threshold_percent) / 100.0
    figure = build_chart(rows, events)
    timeline = build_timeline(events, duration)
    summary = build_summary(rows, events, duration)
    details = build_details(rows)
    explanation, summary_data = build_explanation(
        rows, events, duration, threshold
    )
    report_file = create_report_file(
        rows, events, summary_data, explanation
    )
    annotated_video = None
    if make_annotated_video:
        annotated_video = create_annotated_video(
            video_path,
            rows,
            threshold,
            progress,
        )
    progress(1.0, desc="اكتمل التحليل")
    return (
        summary,
        annotated_video,
        explanation,
        figure,
        timeline,
        details,
        report_file,
    )


CSS = """
html, body {
    background:#050b14 !important;
    min-height:100%;
}
.gradio-container {
    direction: rtl;
    max-width: 1450px !important;
    margin: 0 auto !important;
    color:#ffffff !important;
    background: radial-gradient(circle at top right, #102645 0, #07101e 45%, #050b14 100%);
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6,
.gradio-container p,
.gradio-container small,
.gradio-container strong,
.gradio-container b,
.gradio-container label,
.gradio-container li,
.gradio-container .prose,
.gradio-container .prose * {
    color:#ffffff !important;
}
.gradio-container .form {
    background:#0f1b2d !important;
    border-color:#243a57 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container .input-container {
    background:#142238 !important;
    color:#ffffff !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color:#ffffff !important;
    opacity:.82 !important;
}
.gradio-container .tabs button {
    color:#ffffff !important;
    font-weight:700 !important;
    opacity:.76;
}
.gradio-container .tabs button.selected {
    color:#ffffff !important;
    opacity:1;
}
.gradio-container .plot-container,
.gradio-container .js-plotly-plot,
.gradio-container .plotly {
    width:100% !important;
    background:#09111f !important;
}
.hero {
    padding: 28px;
    border: 1px solid #1f3f66;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(16,39,70,.96), rgba(8,19,35,.96));
    margin-bottom: 18px;
}
.hero h1 { margin:0; color:#f8fafc; font-size:32px; }
.hero p { margin:8px 0 0; color:#ffffff; font-size:16px; }
.status-strip {
    display:flex; gap:10px; flex-wrap:wrap; margin-top:16px;
}
.status-pill {
    color:#ffffff; background:rgba(31,86,145,.26); border:1px solid #285c91;
    padding:7px 12px; border-radius:999px; font-size:12px;
}
.section-title {
    color:#f8fafc; font-size:20px; font-weight:800; margin:12px 0 4px;
}
.summary-grid {
    display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin:8px 0 18px;
}
.metric-card {
    padding:18px; border-radius:14px; background:#0f1b2d; border:1px solid #203653;
}
.metric-card small { display:block; color:#ffffff; margin-bottom:7px; }
.metric-card b { color:#f8fafc; font-size:19px; }
.timeline-panel {
    padding:20px; background:#0f1b2d; border:1px solid #203653; border-radius:16px;
}
.timeline-heading { color:#f8fafc; font-size:18px; font-weight:700; margin-bottom:17px; }
.timeline-track { height:18px; background:#243247; border-radius:12px; position:relative; overflow:hidden; }
.timeline-event { height:100%; position:absolute; top:0; border-radius:10px; min-width:5px; }
.timeline-labels { display:flex; justify-content:space-between; color:#ffffff; font-size:12px; margin-top:6px; }
.event-list { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; margin-top:18px; }
.event-card { padding:12px; border-radius:10px; display:flex; flex-direction:column; gap:4px; color:#f8fafc; }
.violence-event { background:rgba(239,68,68,.13); border:1px solid rgba(239,68,68,.5); }
.weapon-event { background:rgba(245,158,11,.13); border:1px solid rgba(245,158,11,.5); }
.event-card small,.event-card span { color:#ffffff; }
.safe-card { display:flex; gap:12px; align-items:center; color:#d7fbe4; }
.safe-card span { color:#ffffff; }
.safe-icon {
    width:38px; height:38px; display:grid; place-items:center; border-radius:50%;
    background:rgba(34,197,94,.16); color:#22c55e; font-size:20px; font-weight:bold;
}
.analysis-report {
    background:linear-gradient(155deg,#101d31,#0b1423);
    border:1px solid #243a57; border-radius:18px; padding:24px;
    color:#ffffff; direction:rtl; line-height:1.8;
}
.analysis-report,
.analysis-report * {
    color:#ffffff !important;
}
.report-head { display:flex; justify-content:space-between; gap:18px; align-items:flex-start; }
.report-eyebrow { color:#ffffff; font-size:11px; font-weight:800; letter-spacing:.08em; }
.report-head h2 { color:#f8fafc; margin:4px 0; font-size:25px; }
.report-head p { color:#ffffff; margin:0; }
.risk-badge {
    flex:0 0 120px; text-align:center; padding:13px; border-radius:13px;
    background:color-mix(in srgb,var(--risk-color) 12%,transparent);
    border:1px solid var(--risk-color);
}
.risk-badge small { display:block; color:#ffffff; }
.risk-badge b { display:block; color:#ffffff; font-size:21px; }
.report-kpis {
    display:grid; grid-template-columns:repeat(4,1fr); gap:9px; margin:20px 0;
}
.report-kpis > div {
    background:#142238; border:1px solid #253b5a; padding:13px; border-radius:11px;
}
.report-kpis small { display:block; color:#ffffff; }
.report-kpis b { display:block; color:#fff; font-size:17px; margin-top:3px; }
.report-section {
    margin-top:18px; padding-top:16px; border-top:1px solid #233650;
}
.report-section-title { color:#f8fafc; font-size:17px; font-weight:800; margin-bottom:7px; }
.report-section p { color:#ffffff !important; margin:0; }
.duration-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:9px; margin-top:17px; }
.duration-item { padding:12px; border-radius:10px; display:flex; justify-content:space-between; }
.normal-duration { background:#22c55e18; color:#ffffff; border:1px solid #22c55e88; }
.violence-duration { background:#ef444418; color:#ffffff; border:1px solid #ef444488; }
.weapon-duration { background:#f59e0b18; color:#ffffff; border:1px solid #f59e0b88; }
.report-events { display:grid; gap:9px; }
.report-event {
    display:grid; grid-template-columns:70px 1fr 1fr 1fr; gap:10px; align-items:center;
    padding:11px 13px; background:#142238; border-right:4px solid; border-radius:9px;
}
.report-event-number { color:#ffffff; font-size:12px; }
.report-event-class { color:#fff; font-weight:800; }
.report-event-time,.report-event-confidence { color:#ffffff; }
.report-safe { display:flex; align-items:center; gap:10px; color:#dff9e8; }
.report-safe small { display:block; color:#ffffff; }
.report-safe-icon {
    display:grid; place-items:center; width:36px; height:36px; border-radius:50%;
    background:#22c55e22; color:#22c55e; font-weight:900;
}
.report-notice {
    margin-top:20px; background:#f59e0b12; border:1px solid #f59e0b44;
    border-right:4px solid #f59e0b; padding:13px; border-radius:9px; color:#ffffff;
}
#details-table table th,
#details-table table td,
#details-table table th *,
#details-table table td * {
    color:#111827 !important;
    opacity:1 !important;
    font-weight:600;
}
#details-table table thead th {
    background:#dbe3ee !important;
    color:#0f172a !important;
    font-weight:800;
}
#details-table table tbody tr:nth-child(odd) td {
    background:#f8fafc !important;
}
#details-table table tbody tr:nth-child(even) td {
    background:#e9eef5 !important;
}
@media (max-width:850px) { .summary-grid { grid-template-columns:repeat(2,1fr); } }
@media (max-width:850px) {
    .report-kpis { grid-template-columns:repeat(2,1fr); }
    .duration-grid { grid-template-columns:1fr; }
    .report-event { grid-template-columns:1fr 1fr; }
}
"""


with gr.Blocks(
    title="SCVD Intelligence Dashboard",
) as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>SCVD Intelligence Dashboard</h1>
            <p>منصة تحليل زمني لمشاهد العنف والعنف المسلح في فيديوهات المراقبة</p>
            <div class="status-strip">
                <span class="status-pill">SCVD CNN + LSTM</span>
                <span class="status-pill">Person Detection</span>
                <span class="status-pill">Temporal Analysis</span>
                <span class="status-pill">Arabic Explainability</span>
            </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.HTML('<div class="section-title">مصدر الفيديو</div>')
            video_input = gr.Video(label="الفيديو الأصلي", height=430)
            if os.path.exists("test.mp4"):
                gr.Examples(
                    examples=[["test.mp4"]],
                    inputs=[video_input],
                    label="فيديو اختبار جاهز — اضغط لتحميله",
                )
        with gr.Column(scale=2):
            gr.HTML('<div class="section-title">إعدادات التحليل</div>')
            window_input = gr.Slider(
                1.0, 5.0, value=2.0, step=0.5, label="طول كل مقطع (ثانية)"
            )
            stride_input = gr.Slider(
                0.5, 3.0, value=1.0, step=0.5, label="خطوة التحليل (ثانية)"
            )
            threshold_input = gr.Slider(
                40, 90, value=55, step=1, label="حد التنبيه (%)"
            )
            annotated_input = gr.Checkbox(
                value=True,
                label="إنشاء فيديو بالبوكسات",
                info="يستخدم YOLO لتحديد الأشخاص داخل مشاهد التنبيه",
            )
            gr.Markdown(
                "> البوكس يحدد الشخص الموجود في مشهد التنبيه، "
                "ولا يقرر من هو المعتدي."
            )
            analyze_button = gr.Button(
                "تشغيل التحليل الكامل", variant="primary", size="lg"
            )

    summary_output = gr.HTML()
    with gr.Tabs():
        with gr.Tab("مركز النتائج"):
            with gr.Row():
                with gr.Column(scale=5):
                    annotated_video_output = gr.Video(
                        label="الفيديو بعد إضافة التنبيهات والبوكسات",
                        height=430,
                    )
                with gr.Column(scale=3):
                    explanation_output = gr.HTML(
                        '<div class="analysis-report">سيظهر تقرير التحليل هنا بعد انتهاء المعالجة.</div>'
                    )
                    report_output = gr.File(
                        label="تنزيل تقرير التحليل المنسق HTML"
                    )
        with gr.Tab("الرسم البياني"):
            chart_output = gr.Plot(show_label=False)
        with gr.Tab("الخط الزمني"):
            timeline_output = gr.HTML()
        with gr.Tab("التحليل التفصيلي"):
            details_output = gr.Dataframe(
                interactive=False,
                wrap=True,
                show_search="search",
                elem_id="details-table",
            )

    analyze_button.click(
        fn=analyze_video,
        inputs=[
            video_input,
            window_input,
            stride_input,
            threshold_input,
            annotated_input,
        ],
        outputs=[
            summary_output,
            annotated_video_output,
            explanation_output,
            chart_output,
            timeline_output,
            details_output,
            report_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
