import os
import tempfile

import streamlit as st

from SCVD_Professional_GUI import (
    CLASS_AR,
    CLASSES,
    build_chart,
    build_details,
    build_explanation,
    build_timeline,
    create_annotated_video,
    create_report_file,
    find_events,
    format_time,
    predict_windows,
    smooth_probabilities,
)


st.set_page_config(
    page_title="SCVD Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        background:
            radial-gradient(circle at top right, #102645 0, #07101e 44%, #050b14 100%);
        color: #fff;
    }
    [data-testid="stSidebar"] {
        background: #0b1423;
        border-left: 1px solid #243a57;
    }
    [data-testid="stSidebar"] * { color: #fff !important; }
    .hero {
        padding: 27px 30px;
        border: 1px solid #254263;
        border-radius: 20px;
        background: linear-gradient(135deg, #122943, #091525);
        margin-bottom: 18px;
    }
    .hero h1 { color:#fff; font-size:32px; margin:0; }
    .hero p { color:#d6e5f7; margin:7px 0 0; }
    .pill {
        display:inline-block; margin:14px 0 0 7px; padding:6px 11px;
        border:1px solid #32699f; border-radius:999px; color:#fff;
        background:#1c4b752c; font-size:12px;
    }
    .metrics {
        display:grid; grid-template-columns:repeat(4,1fr);
        gap:12px; margin:14px 0 20px;
    }
    .metric {
        background:#101d31; border:1px solid #263e5d;
        border-radius:14px; padding:17px;
    }
    .metric small { display:block; color:#cbd8e8; margin-bottom:7px; }
    .metric b { color:#fff; font-size:20px; }
    .analysis-report, .analysis-report * { color:#fff !important; }
    .analysis-report {
        background:linear-gradient(155deg,#101d31,#0b1423);
        border:1px solid #243a57; border-radius:18px;
        padding:24px; line-height:1.8;
    }
    .report-head { display:flex; justify-content:space-between; gap:18px; }
    .report-head h2 { margin:4px 0; }
    .report-eyebrow { font-size:11px; font-weight:800; }
    .risk-badge {
        min-width:110px; text-align:center; padding:12px;
        border:1px solid var(--risk-color); border-radius:12px;
    }
    .risk-badge small,.risk-badge b { display:block; }
    .report-kpis {
        display:grid; grid-template-columns:repeat(4,1fr);
        gap:9px; margin:18px 0;
    }
    .report-kpis > div,.report-event {
        background:#142238; border:1px solid #253b5a;
        padding:12px; border-radius:10px;
    }
    .report-kpis small,.report-kpis b { display:block; }
    .report-section {
        margin-top:17px; padding-top:14px; border-top:1px solid #233650;
    }
    .report-section-title { font-size:17px; font-weight:800; }
    .duration-grid {
        display:grid; grid-template-columns:repeat(3,1fr);
        gap:9px; margin-top:16px;
    }
    .duration-item {
        padding:12px; border-radius:10px;
        display:flex; justify-content:space-between;
    }
    .normal-duration { background:#22c55e18; border:1px solid #22c55e88; }
    .violence-duration { background:#ef444418; border:1px solid #ef444488; }
    .weapon-duration { background:#f59e0b18; border:1px solid #f59e0b88; }
    .report-events { display:grid; gap:9px; }
    .report-event {
        display:grid; grid-template-columns:70px 1fr 1fr 1fr;
        gap:10px; border-right:4px solid;
    }
    .report-notice {
        margin-top:18px; background:#f59e0b12; border:1px solid #f59e0b44;
        border-right:4px solid #f59e0b; padding:13px; border-radius:9px;
    }
    .timeline-panel {
        padding:20px; background:#0f1b2d;
        border:1px solid #203653; border-radius:16px; color:#fff;
    }
    .timeline-heading { font-size:18px; font-weight:800; margin-bottom:15px; }
    .timeline-track {
        height:18px; background:#243247; border-radius:12px;
        position:relative; overflow:hidden;
    }
    .timeline-event { height:100%; position:absolute; top:0; border-radius:10px; }
    .timeline-labels { display:flex; justify-content:space-between; margin-top:6px; }
    .event-list {
        display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
        gap:10px; margin-top:16px;
    }
    .event-card { padding:12px; border-radius:10px; display:flex; flex-direction:column; }
    .violence-event { background:#ef444420; border:1px solid #ef444480; }
    .weapon-event { background:#f59e0b20; border:1px solid #f59e0b80; }
    .safe-card { display:flex; gap:11px; align-items:center; margin-top:15px; }
    .safe-card div { display:flex; flex-direction:column; }
    .safe-icon {
        display:grid; place-items:center; width:36px; height:36px;
        border-radius:50%; background:#22c55e24; color:#4ade80;
        font-size:19px; font-weight:900;
    }
    div[data-testid="stDataFrame"] {
        background:#fff; border-radius:12px; overflow:hidden;
    }
    @media(max-width:850px) {
        .metrics,.report-kpis { grid-template-columns:repeat(2,1fr); }
        .duration-grid { grid-template-columns:1fr; }
    }
    </style>
    <div class="hero">
      <h1>SCVD Intelligence Dashboard</h1>
      <p>تحليل زمني ذكي لمشاهد العنف والعنف المسلح في فيديوهات المراقبة</p>
      <span class="pill">CNN + LSTM</span>
      <span class="pill">Temporal Analysis</span>
      <span class="pill">Person Detection</span>
      <span class="pill">Arabic Explainability</span>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("إعدادات التحليل")
    window_seconds = st.slider(
        "طول كل مقطع (ثانية)", 1.0, 5.0, 2.0, 0.5
    )
    stride_seconds = st.slider(
        "خطوة التحليل (ثانية)", 0.5, 3.0, 1.0, 0.5
    )
    alert_threshold_percent = st.slider(
        "حد التنبيه (%)", 40, 90, 55, 1
    )
    make_boxes = st.checkbox(
        "إنشاء فيديو بالبوكسات",
        value=False,
        help="قد يستغرق وقتًا إضافيًا ويستهلك ذاكرة أكبر.",
    )
    st.info(
        "البوكس يحدد الأشخاص الموجودين في مشهد التنبيه، "
        "ولا يثبت أن الشخص هو المعتدي."
    )


source_choice = st.radio(
    "مصدر الفيديو",
    ["فيديو الاختبار الجاهز", "رفع فيديو جديد"],
    horizontal=True,
)
uploaded_file = None
video_path = None

if source_choice == "فيديو الاختبار الجاهز":
    ready_videos = {
        "فيديو الاختبار الأول": "test.mp4",
        "فيديو الاختبار الثاني": "test2.mp4",
    }
    available_videos = {
        label: path
        for label, path in ready_videos.items()
        if os.path.exists(path)
    }
    if available_videos:
        selected_video = st.selectbox(
            "اختر فيديو جاهزًا",
            options=list(available_videos.keys()),
        )
        video_path = available_videos[selected_video]
        st.video(video_path)
    else:
        st.error("لا توجد فيديوهات اختبار جاهزة في المستودع.")
else:
    uploaded_file = st.file_uploader(
        "ارفع فيديو المراقبة",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded_file is not None:
        st.video(uploaded_file)


if st.button(
    "تشغيل التحليل الكامل",
    type="primary",
    use_container_width=True,
):
    if stride_seconds > window_seconds:
        st.error("خطوة التحليل يجب ألا تكون أكبر من طول المقطع.")
        st.stop()
    if source_choice == "رفع فيديو جديد":
        if uploaded_file is None:
            st.error("ارفع فيديو أولًا.")
            st.stop()
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            video_path = temp_video.name

    status = st.status("جاري تحليل الفيديو...", expanded=True)
    try:
        status.write("تقسيم الفيديو إلى مقاطع زمنية")
        rows, duration = predict_windows(
            video_path,
            float(window_seconds),
            float(stride_seconds),
        )
        status.write("تنعيم الاحتمالات ودمج الأحداث المتتابعة")
        rows = smooth_probabilities(rows)
        threshold = float(alert_threshold_percent) / 100.0
        events = find_events(rows, threshold)
        explanation, summary_data = build_explanation(
            rows, events, duration, threshold
        )
        report_path = create_report_file(
            rows, events, summary_data, explanation
        )
        annotated_path = None
        if make_boxes:
            status.write("اكتشاف الأشخاص وإنشاء الفيديو المعلّم")
            annotated_path = create_annotated_video(
                video_path, rows, threshold
            )
        st.session_state["analysis"] = {
            "rows": rows,
            "duration": duration,
            "events": events,
            "explanation": explanation,
            "summary": summary_data,
            "report_path": report_path,
            "annotated_path": annotated_path,
        }
        status.update(
            label="اكتمل التحليل بنجاح",
            state="complete",
            expanded=False,
        )
    except Exception as error:
        status.update(label="فشل التحليل", state="error")
        st.exception(error)


if "analysis" in st.session_state:
    result = st.session_state["analysis"]
    events = result["events"]
    rows = result["rows"]
    duration = result["duration"]
    overall_class = max(
        CLASSES,
        key=lambda name: sum(row[name] for row in rows) / len(rows),
    )
    first_threat = format_time(events[0]["start"]) if events else "لا يوجد"
    status_text = "تم رصد تهديد" if events else "الفيديو آمن"

    st.markdown(
        f"""
        <div class="metrics">
          <div class="metric"><small>الحالة العامة</small><b>{status_text}</b></div>
          <div class="metric"><small>التصنيف الغالب</small><b>{CLASS_AR[overall_class]}</b></div>
          <div class="metric"><small>بداية أول تهديد</small><b>{first_threat}</b></div>
          <div class="metric"><small>مدة الفيديو</small><b>{format_time(duration)}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_results, tab_chart, tab_timeline, tab_details = st.tabs(
        ["مركز النتائج", "الرسم البياني", "الخط الزمني", "التحليل التفصيلي"]
    )
    with tab_results:
        left, right = st.columns([1.15, 1])
        with left:
            if result["annotated_path"]:
                st.subheader("الفيديو بعد إضافة التنبيهات والبوكسات")
                st.video(result["annotated_path"])
            else:
                st.subheader("الفيديو محل التحليل")
                st.video(video_path)
        with right:
            st.markdown(
                result["explanation"],
                unsafe_allow_html=True,
            )
            with open(result["report_path"], "rb") as report_file:
                st.download_button(
                    "تنزيل تقرير التحليل HTML",
                    data=report_file.read(),
                    file_name="SCVD_analysis_report.html",
                    mime="text/html",
                    use_container_width=True,
                )

    with tab_chart:
        st.plotly_chart(
            build_chart(rows, events),
            use_container_width=True,
        )
    with tab_timeline:
        st.markdown(
            build_timeline(events, duration),
            unsafe_allow_html=True,
        )
    with tab_details:
        st.dataframe(
            build_details(rows),
            use_container_width=True,
            hide_index=True,
            height=560,
        )
