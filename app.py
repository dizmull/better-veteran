# BetterVA for the Congressional App Challenge :)
# Requires: gradio>=5, transformers, torch, torchvision, pillow
# Start:    python app.py
# by Uchenna Uduma

from __future__ import annotations
import gradio as gr
import torch
from PIL import Image
from typing import List, Dict, Tuple
from transformers import AutoImageProcessor, AutoModelForImageClassification

APP_NAME = "BetterVA"
MODEL_ID = "DifeiT/rsna-intracranial-hemorrhage-detection"
RISK_THRESHOLD = 0.50  # demo banner threshold

# --------------------------- THEME / CSS ---------------------------
CSS = """
:root{
  --accent:#EAF6FF;       /* SKY */
  --accent-2:#EAF6FF;
  --text:#0b1420;
  --line:#e9eef2;
}

*{color:var(--text);}
.gradio-container{
  background:#fff !important;
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','SF Pro Display','Segoe UI',Roboto,Arial,Helvetica,'Apple Color Emoji','Segoe UI Emoji';
  font-size:16px;
}
footer,.footer,#footer{display:none !important;} /* hide gradio footer */

.header{padding:12px 16px;border-bottom:1px solid var(--line);font-weight:800;text-align:center;}
.layout{display:flex;gap:16px;align-items:stretch;}
.sidebar{
  width:250px;min-width:220px;
  border-right:1px solid var(--line);
  padding:12px;border-radius:18px;
  background:#F7FAFE;
  height:calc(100vh - 120px);
  position:sticky;top:88px;overflow:auto;
}
.nav button{
  width:100% !important;justify-content:flex-start !important;
  background:#fff !important;border:1px solid var(--line) !important;
  color:var(--text) !important;border-radius:12px !important;
  padding:12px !important;margin-bottom:10px !important;font-weight:800 !important;
}
.nav button:hover{background:#f5fbff !important;}
.content{flex:1;padding:4px;}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px;}
.cta{display:flex;gap:16px;flex-wrap:wrap;justify-content:center;}
.cta-card{
  flex:1 1 360px;max-width:560px;
  background:#fff;border:1px solid var(--line);
  border-radius:20px;padding:22px;text-align:center;
}
.cta-card h2{margin:0 0 8px 0;font-size:22px;}
.cta-card p{opacity:.85;margin:0 0 14px 0;}
.cta-card button{
  background:var(--accent) !important;color:#000 !important;
  border:none !important;border-radius:12px !important;
  padding:12px 18px !important;font-weight:800 !important;
}
.btn,button{
  background:var(--accent) !important;color:#000 !important;
  border:none !important;border-radius:12px !important;
  padding:10px 16px !important;font-weight:800 !important;
}
button:hover{filter:brightness(.97);}
.df table,.df thead tr,.df tbody tr,.df td,.df th{
  background:#fff !important;color:var(--text) !important;border-color:var(--line) !important;
}
.df th{font-weight:800;}
.scan-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:960px){.scan-grid{grid-template-columns:1fr;}}
.result-banner{color:#C23B3B;font-size:22px;font-weight:800;margin-bottom:6px;}
.result-banner-two{color:#7EF28F;font-size:22px;font-weight:800;margin-bottom:6px;}
.small-note{font-size:12px;opacity:.85;}
.input-light *{color:var(--text) !important;}
.header-title{font-size:20px;font-weight:800;}

/* ---------- Startup Overlay ---------- */
#overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.35);
  display:flex;align-items:center;justify-content:center;z-index:9999;
}
#overlay .panel{
  background:#fff;border:1px solid var(--line);
  border-radius:16px;padding:18px;max-width:520px;width:92%;
}
#overlay .row{display:flex;gap:10px;}
#overlay .row > div{flex:1;}
#overlay label{font-weight:700;margin-bottom:6px;display:block;}
#overlay .warn{margin-top:8px;color:#b00020;font-weight:700;display:none;}

/* ---------- PCL STYLING ---------- */

/* Light PCL container */
.pcl {
  background: #F5FBFF !important;
  padding: 16px !important;
  border-radius: 16px !important;
}

/* Remove dark gradio block defaults */
.pcl .gr-form, 
.pcl .gr-block 
  background: transparent !important;
  border: none !important;
}

/* PCL text (questions + instructions) */
.pcl .gr-markdown, 
.pcl .gr-markdown * {
  color: #0b1420 !important;
}

/* Progress text */
.pcl .progress-text, 
.pcl .progress-text * {
  color: #0b1420 !important;
}

/* Radio buttons */
.pcl .gr-radio, 
.pcl .gr-radio-group label {
  background: #E8F3FF !important;
  color: #0b1420 !important;
  border: 1px solid #A7D2FF !important;
  border-radius: 10px !important;
  padding: 10px 12px !important;
  font-weight: 600 !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  margin-bottom: 6px !important;
}

/* Hover state */
.pcl .gr-radio-group label:hover {
  background: #D8EDFF !important;
  border-color: #7CC2FF !important;
}

/* Selected */
.pcl .gr-radio-group input[type="radio"]:checked + label {
  background: #92C8FF !important;
  border: 2px solid #2B8FFF !important;
  color: #000 !important;
  font-weight: 700 !important;
}

/* Keep cards white */
.card,
.cta-card,
.sidebar {
  background: #fff !important;
}
"""


DISCLAIMER = ("This website uses a Machine Learning model to detect Intracranial Hemorrhages from CT images. An intracranial hemorrhage is bleeding inside the skull, which can occur in the brain tissue itself or in the spaces surrounding it. This is a life-threatening condition, often caused by a ruptured blood vessel due to high blood pressure, injury, or other medical issues. The bleeding creates pressure on the brain, which disrupts oxygen supply and requires immediate medical attention. ")

# --------------------------- MODEL ---------------------------
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
except TypeError:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

def _softmax(x: torch.Tensor) -> torch.Tensor:
    e = torch.exp(x - x.max())
    return e / e.sum()

ALLOWED_SUBTYPES = {"epidural","subdural","subarachnoid","intraparenchymal","intraventricular","any"}

def infer_brain_scan(img: Image.Image) -> Tuple[List[Dict], float, str]:
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = _softmax(logits).tolist()
    id2label = model.config.id2label
    labeled = [{"label": id2label[i], "prob": float(p)} for i,p in enumerate(probs)]
    filtered = []
    for d in labeled:
        lbl = d["label"].lower()
        if any(k in lbl for k in ALLOWED_SUBTYPES):
            filtered.append({"label": d["label"], "prob": d["prob"]})
    filtered.sort(key=lambda x: x["prob"], reverse=True)
    max_prob = filtered[0]["prob"] if filtered else 0.0
    flag = "High" if max_prob >= RISK_THRESHOLD else "Low"
    return filtered, max_prob, flag

def rows_for_table(subtypes: List[Dict]):
    return [[d["label"], round(d["prob"], 4)] for d in subtypes[:6]]

# --------------------------- PCL-5 DATA ---------------------------
PCL5_QUESTIONS = [
    'Repeated, disturbing, and unwanted memories of the stressful experience?',
    'Repeated, disturbing dreams of the stressful experience?',
    'Suddenly feeling or acting as if the stressful experience were actually happening again?',
    'Feeling very upset when something reminded you of the stressful experience?',
    'Having strong physical reactions when reminded (e.g., heart pounding, sweating)?',
    'Avoiding memories, thoughts, or feelings related to the stressful experience?',
    'Avoiding external reminders (people, places, conversations, activities)?',
    'Trouble remembering important parts of the stressful experience?',
    'Strong negative beliefs about yourself, others, or the world?',
    'Blaming yourself or someone else for the stressful experience or what followed?',
    'Strong negative feelings such as fear, horror, anger, guilt, or shame?',
    'Loss of interest in activities you used to enjoy?',
    'Feeling distant or cut off from other people?',
    'Trouble experiencing positive feelings?',
    'Irritable behavior, angry outbursts, or acting aggressively?',
    'Taking too many risks or doing harmful things?',
    'Being “superalert,” watchful, or on guard?',
    'Feeling jumpy or easily startled?',
    'Difficulty concentrating?',
    'Trouble falling or staying asleep?'
]
PCL5_CHOICES = [
    ("0 - Not at all", 0),
    ("1 - A little bit", 1),
    ("2 - Moderately", 2),
    ("3 - Quite a bit", 3),
    ("4 - Extremely", 4),
]
def pcl_init():
    return {"idx":0, "answers":[None]*len(PCL5_QUESTIONS), "started":False, "done":False}
def pcl_score(ans): return sum(a or 0 for a in ans)
def pcl_progress(s): 
    n_done = sum(1 for a in s["answers"] if a is not None)
    

# --------------------------- APP ---------------------------
with gr.Blocks(css=CSS, title=APP_NAME, fill_height=True) as demo:
    # Header
    gr.Markdown(f"<div class='header header-title'>{APP_NAME}</div>")

    # ---------- STARTUP OVERLAY (stable, no JS) ----------
    

    # ---------- MAIN LAYOUT ----------
    with gr.Row(elem_classes="layout"):
        # Sidebar
        with gr.Column(elem_classes="sidebar"):
            gr.Markdown("#### Go to:")
            with gr.Column(elem_classes="nav"):
                home_btn = gr.Button("Home")
                scan_btn = gr.Button("Hemorrhage Detect")
                pcl_btn  = gr.Button("PCL-5 Practice Test", interactive=True)  #always on
                res_btn  = gr.Button("Resources")

        # Content panels
        with gr.Column(elem_classes="content"):
            # HOME
            home_panel = gr.Group(visible=True)
            with home_panel:
                welcome_md = gr.Markdown("")
                with gr.Row(elem_classes="cta"):
                    with gr.Column(elem_classes="cta-card"):
                        gr.HTML("<h2>Let’s take a PCL-5 Mock Test!</h2><p>About 10–15 minutes. Practice to prepare for the real deal, and take the next steps to treating possible PTSD!</p>")
                        home_to_pcl = gr.Button("Start MPCL-5", interactive=True)  #always on
                    with gr.Column(elem_classes="cta-card"):
                        gr.HTML("<h2>Or, Let’s take a Brain Scan!</h2><p>Upload a CT image (JPG/PNG) to see if you may be at risk of a Brain Bleed.</p>")
                        home_to_scan = gr.Button("Start Brain Scan")
                with gr.Column(elem_classes="card"):
                    gov_btn = gr.Button("Access VA Support & Benefits! (VA.gov)")

            # SCAN
            scan_panel = gr.Group(visible=False)
            with scan_panel:
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### **Upload a Brain Scan Image**")
                    gr.Markdown(DISCLAIMER)
                with gr.Column(elem_classes="card"):
                    with gr.Row(elem_classes="scan-grid"):
                        with gr.Column():
                            img_in = gr.Image(type="pil", label="Upload Image (JPG/PNG)", height=400, elem_classes="input-light")
                            run_btn = gr.Button("Analyze my Brain Scan!")
                        with gr.Column():
                            result_md = gr.Markdown()
                            table_df = gr.Dataframe(
                                headers=["Hemorrhage Type","Confidence"],
                                datatype=["str","number"],
                                row_count=(6,"dynamic"),
                                col_count=(2,"fixed"),
                                interactive=False,
                                elem_classes="df"
                            )
                            hidden_risk = gr.Textbox(visible=False)

                def do_scan(image):
                    if image is None:
                        return ("Please upload an image of a Brain Scan!", [], "Low")
                    subs, maxp, flag = infer_brain_scan(image)
                    table = rows_for_table(subs)
                    if flag == "High":
                        msg = ("<div class='result-banner'>You may be at possible risk for an intracranial hemorrhage.</div>"
                               "<div class='small-note'>*This is not a diagnosis. Please discuss with a clinician.</div>")
                    else:
                        msg = ("<div class='result-banner-two'>Hooray! No Intracranial Hemorrhage was detected!</div>"
                               "<div class='small-note'>*This is not a diagnosis. Please discuss with a clinician.</div>")
                    return msg, table, flag

                run_btn.click(do_scan, [img_in], [result_md, table_df, hidden_risk])

                # mobile vibration on High
                hidden_risk.change(None, [hidden_risk], [], js="""
                    (v)=>{ try{ if(v==='High' && 'vibrate' in navigator){ navigator.vibrate([200,100,200]); } }catch(e){} }
                """)

            # PCL
            pcl_panel = gr.Group(visible=False, elem_classes="pcl")
            with pcl_panel, gr.Column(elem_classes="pcl"):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("<div style='font-size:18px;font-weight:800;'>PCL-5 Mock Test "
                                "<span title='Please answer based on symptoms felt in THE PAST MONTH.' style='font-weight:900;padding:0 8px;cursor:help;'>ⓘ</span>"
                                "</div>")
                    gr.Markdown("Estimated time: **10–15 minutes**. You got this!\nThe PCL-5 is a 20-item self-report questionnaire, the PTSD Checklist for DSM-5, that assesses the presence and severity of symptoms of post-traumatic stress disorder (PTSD). It is used to monitor symptom severity, screen individuals for PTSD, and assist in making a provisional diagnosis, though a formal diagnosis requires clinical interviews. The questions align with the criteria for PTSD in the DSM-5 and cover symptom clusters like re-experiencing, avoidance, negative changes in cognition/mood, and changes in arousal/reactivity. ")
                # Intro
                intro = gr.Group(visible=True)
                with intro:
                    start_pcl = gr.Button("Start Practice Test!")
                # Test
                test = gr.Group(visible=False)
                with test:
                    pcl_state = gr.State(pcl_init())
                    progress = gr.Markdown("Progress: 0/20")
                    q_text = gr.Markdown("Question", elem_classes="pcl")
                    choice = gr.Radio([c[0] for c in PCL5_CHOICES], label="How much have you been bothered by this?", value=None)
                    with gr.Row():
                        back_btn = gr.Button("Back")
                        next_btn = gr.Button("Next")
                        restart_btn = gr.Button("Restart")
                    done_msg = gr.Markdown(visible=False)

                def load_q(s:Dict):
                    i = s["idx"]
                    txt = f"**Q{i+1}/{len(PCL5_QUESTIONS)}**  {PCL5_QUESTIONS[i]}"
                    prog = pcl_progress(s)
                    val = None
                    if s["answers"][i] is not None:
                        for lab,num in PCL5_CHOICES:
                            if num == s["answers"][i]:
                                val = lab
                                break
                    return prog, txt, val

                def begin_test(s:Dict):
                    s = pcl_init(); s["started"]=True
                    return s, gr.update(visible=False), gr.update(visible=True), *load_q(s), gr.update(visible=False)
                start_pcl.click(begin_test, [pcl_state], [pcl_state, intro, test, progress, q_text, choice, done_msg])

                def on_choice(label:str|None, s:Dict):
                    i = s["idx"]
                    if label is not None:
                        for lab,num in PCL5_CHOICES:
                            if lab==label: s["answers"][i]=num; break
                    return s, *load_q(s)
                choice.change(on_choice, [choice, pcl_state], [pcl_state, progress, q_text, choice])

                def go_next(s:Dict):
                    i=s["idx"]
                    if s["answers"][i] is None:
                        return s, *load_q(s), gr.update(visible=False)
                    if i < len(PCL5_QUESTIONS)-1:
                        s["idx"] += 1
                        return s, *load_q(s), gr.update(visible=False)
                    s["done"]=True
                    total = pcl_score(s["answers"])
                    msg = f"**Completed.** Total score: **{total}**. This is not a diagnosis; discuss results with a clinician."
                    if total >= 33:
                        msg += " Your results show a high risk of PTSD symptoms. If this is a mistake, take the Mock exam again!"
                    elif total >= 21:
                        msg += " Your results show a moderate risk of PTSD symptoms. Consider letting a clinician know!"
                    else:
                        msg += " Your results show a low risk of PTSD symptoms. But, you're never too safe to check in with a neurologist!"

                    return s, *load_q(s), gr.update(value=msg, visible=True)
                next_btn.click(go_next, [pcl_state], [pcl_state, progress, q_text, choice, done_msg])

                def go_back(s:Dict):
                    if s["idx"]>0: s["idx"]-=1
                    return s, *load_q(s), gr.update(visible=False)
                back_btn.click(go_back, [pcl_state], [pcl_state, progress, q_text, choice, done_msg])

                def ask_restart(s:Dict):
                    if any(a is not None for a in s["answers"]):
                        return gr.update(visible=False), gr.update(visible=True)
                    s2=pcl_init(); s2["started"]=True
                    return gr.update(visible=True), gr.update(visible=False)
                restart_btn.click(ask_restart, [pcl_state], [test, intro])  # simple confirm-free toggle

            # RESOURCES
            res_panel = gr.Group(visible=False)
            with res_panel:
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Veteran Resources Near You")
                    gr.Markdown("Use your location or enter a ZIP to open Google Maps with nearby options.")
                    with gr.Row():
                        use_loc = gr.Button("Use my location")
                        zip_in = gr.Textbox(label="Or ZIP code", placeholder="e.g., 08234")
                        apply_zip = gr.Button("Search")
                    lat = gr.Textbox(visible=False); lon = gr.Textbox(visible=False)
                results = gr.Markdown(elem_classes="card")

                def build_from_latlon(lat_v,lon_v):
                    if lat_v and lon_v:
                        q=f"{lat_v},{lon_v}"
                        return ("#### Quick Links (based on your location)\n"
                                f"- [VA facilities near you](https://www.google.com/maps/search/VA+facility/@{q},12z)\n"
                                f"- [Vet Centers near you](https://www.google.com/maps/search/Vet+Center/@{q},12z)\n"
                                f"- [Veterans services near you](https://www.google.com/maps/search/Veterans+services/@{q},12z)\n"
                                "\n*Opens Google Maps.*")
                    return ("#### Quick Links\n"
                            "- [VA facilities near me](https://www.google.com/maps/search/VA+facility+near+me)\n"
                            "- [Vet Centers near me](https://www.google.com/maps/search/Vet+Center+near+me)\n"
                            "- [Veterans services near me](https://www.google.com/maps/search/Veterans+services+near+me)\n"
                            "\n*Opens Google Maps.*\n"
                            "### Info Page"
                            "\n- [What is an Intracranial Hemorrhage?](https://my.clevelandclinic.org/health/diseases/14480-brain-bleed-hemorrhage-intracranial-hemorrhage)"
                            "\n- [What is PTSD?](https://www.mayoclinic.org/diseases-conditions/post-traumatic-stress-disorder/symptoms-causes/syc-20355967)"
                            "\n- [What is Traumatic Brain Injury (TBI)?](https://www.cdc.gov/traumaticbraininjury/index.html)\n"
                            "### Machine Learning and AI\n"
                            "- [How does Medical AI work?](https://www.foreseemed.com/artificial-intelligence-in-healthcare)\n"
                            "- [What is Machine Learning?](https://www.ibm.com/think/topics/machine-learning)"
                            )

                def build_from_zip(z):
                    z=(z or "").strip()
                    if not z: return build_from_latlon("","")
                    return (f"#### Quick Links (ZIP {z})\n"
                            f"- [VA facilities near {z}](https://www.google.com/maps/search/VA+facility+near+{z})\n"
                            f"- [Vet Centers near {z}](https://www.google.com/maps/search/Vet+Center+near+{z})\n"
                            f"- [Veterans services near {z}](https://www.google.com/maps/search/Veterans+services+near+{z})\n"
                            "\n*Opens Google Maps.*")

                demo.load(lambda: build_from_latlon("",""), None, [results])

                use_loc.click(
                    None, [], [lat,lon],
                    js="""
                    () => new Promise((resolve)=>{
                      if(!navigator.geolocation){ resolve(["",""]); return; }
                      navigator.geolocation.getCurrentPosition(
                        (pos)=>resolve([String(pos.coords.latitude), String(pos.coords.longitude)]),
                        (_)=>resolve(["",""])
                      );
                    })
                    """
                )
                lat.change(build_from_latlon, [lat,lon], [results])
                lon.change(build_from_latlon, [lat,lon], [results])
                apply_zip.click(build_from_zip, [zip_in], [results])

    # ---------- NAVIGATION (A1: direct show/hide) ----------
    def nav_to(target:str, adult:bool):
        show_home = target=="home"
        show_scan = target=="scan"
        show_pcl  = target=="pcl"
        show_res  = target=="res"
        return (gr.update(visible=show_home),
                gr.update(visible=show_scan),
                gr.update(visible=show_pcl),
                gr.update(visible=show_res))

    home_btn.click(lambda a: nav_to("home", True), [], [home_panel, scan_panel, pcl_panel, res_panel])
    scan_btn.click(lambda a: nav_to("scan", True), [], [home_panel, scan_panel, pcl_panel, res_panel])
    pcl_btn.click(lambda a: nav_to("pcl", True), [], [home_panel, scan_panel, pcl_panel, res_panel])
    res_btn.click(lambda a: nav_to("res", True), [], [home_panel, scan_panel, pcl_panel, res_panel])
    home_to_scan.click(lambda a: nav_to("scan", True), [], [home_panel, scan_panel, pcl_panel, res_panel])
    home_to_pcl.click(lambda a: nav_to("pcl", True), [], [home_panel, scan_panel, pcl_panel, res_panel])

    # VA.gov (new tab)
    gov_btn.click(None, [], [], js="""()=>{ window.open('https://www.va.gov','_blank'); }""")

# RUN
if __name__ == "__main__":
    demo.launch(share=True)
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
