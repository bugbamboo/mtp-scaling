import numpy as np
import json
import random
import os
import ipywidgets as widgets
from IPython.display import display, HTML

###############################################################################
# 0. Build your list of features (rnd) as before
###############################################################################
indices = json.load(open("random_indices.json"))
controls = {'layer8_neuron': [5829, 176, 7384, 428, 3939],
 'layer12_neuron': [4937, 7160, 11967, 900, 3253],
 'layer16_neuron': [10954, 8425, 12380, 3975, 12420],
 'layer20_neuron': [1874, 7597, 6851, 43, 5394],
 'layer24_neuron': [7654, 8273, 861, 8517, 8707],
 'layer8_sae': [19410, 10654, 17937, 7018, 14683],
 'layer12_sae': [18239, 27605, 20584, 717, 8774],
 'layer16_sae': [28495, 17685, 16190, 25083, 6058],
 'layer20_sae': [9662, 21290, 16746, 23244, 11129],
 'layer24_sae': [4347, 27239, 18830, 18771, 8693]}

tot = []
crtl = []
for x in indices.keys():
    list1 = indices[x]
    list2 = controls[x]
    for q in list1:
        if q not in list2:
            label = x.split("_")[1]
            l = x.split("_")[0]
            tot.append((f"{q}_{label}", f"{label}s/{l}/{q}_image.html", f"{label}s/{l}/{q}_text.html"))
    for q in list2:
        label = x.split("_")[1]
        l = x.split("_")[0]
        crtl.append((f"{q}_{label}", f"{label}s/{l}/{q}_image.html", f"{label}s/{l}/{q}_text.html"))

# Randomly pick 40 from tot and 10 from crtl, then shuffle
rnd = random.sample(tot, 40) + random.sample(crtl, 10)
random.shuffle(rnd)

###############################################################################
# Create the 'annotations' folder if not already there
###############################################################################
os.makedirs('annotations', exist_ok=True)

###############################################################################
# 1. YES/NO Buttons for image explanation
###############################################################################

def reset_buttons_img():
    button_yes_img.button_style = 'success'
    button_no_img.button_style = 'danger'

def dim_other_buttons_img(selected_button):
    reset_buttons_img()
    for b in (button_yes_img, button_no_img):
        if b != selected_button:
            b.button_style = ''

def yes_img(arg):
    nonlocal_val['img_interp'] = 'yes'
    dim_other_buttons_img(button_yes_img)
    img_explanation_text.disabled = False

def no_img(arg):
    nonlocal_val['img_interp'] = 'no'
    dim_other_buttons_img(button_no_img)
    img_explanation_text.disabled = True
    img_explanation_text.value = ''

button_yes_img = widgets.Button(description='Yes', button_style='success')
button_no_img  = widgets.Button(description='No',  button_style='danger')
button_yes_img.on_click(yes_img)
button_no_img.on_click(no_img)

img_yes_no_box = widgets.HBox([button_yes_img, button_no_img])

img_explanation_text = widgets.Textarea(
    placeholder='Enter short explanation for the image feature if "Yes"...',
    disabled=True,
    layout=widgets.Layout(width='400px', height='100px')
)

###############################################################################
# 2. YES/NO Buttons for text explanation
###############################################################################

def reset_buttons_txt():
    button_yes_txt.button_style = 'success'
    button_no_txt.button_style = 'danger'

def dim_other_buttons_txt(selected_button):
    reset_buttons_txt()
    for b in (button_yes_txt, button_no_txt):
        if b != selected_button:
            b.button_style = ''

def yes_txt(arg):
    nonlocal_val['text_interp'] = 'yes'
    dim_other_buttons_txt(button_yes_txt)
    txt_explanation_text.disabled = False

def no_txt(arg):
    nonlocal_val['text_interp'] = 'no'
    dim_other_buttons_txt(button_no_txt)
    txt_explanation_text.disabled = True
    txt_explanation_text.value = ''

button_yes_txt = widgets.Button(description='Yes', button_style='success')
button_no_txt  = widgets.Button(description='No',  button_style='danger')
button_yes_txt.on_click(yes_txt)
button_no_txt.on_click(no_txt)

txt_yes_no_box = widgets.HBox([button_yes_txt, button_no_txt])

txt_explanation_text = widgets.Textarea(
    placeholder='Enter short explanation for the text feature if "Yes"...',
    disabled=True,
    layout=widgets.Layout(width='400px', height='100px')
)

###############################################################################
# 3. YES/NO Buttons for "Do they match?" => is_multimodal
###############################################################################

def reset_buttons_mm():
    button_yes_mm.button_style = 'success'
    button_no_mm.button_style = 'danger'

def dim_other_buttons_mm(selected_button):
    reset_buttons_mm()
    for b in (button_yes_mm, button_no_mm):
        if b != selected_button:
            b.button_style = ''

def yes_mm(arg):
    nonlocal_val['is_multimodal'] = 'yes'
    dim_other_buttons_mm(button_yes_mm)

def no_mm(arg):
    nonlocal_val['is_multimodal'] = 'no'
    dim_other_buttons_mm(button_no_mm)

button_yes_mm = widgets.Button(description='Yes', button_style='success')
button_no_mm  = widgets.Button(description='No',  button_style='danger')
button_yes_mm.on_click(yes_mm)
button_no_mm.on_click(no_mm)

mm_yes_no_box = widgets.HBox([button_yes_mm, button_no_mm])

###############################################################################
# 4. We'll store current states in a dictionary for the callbacks
###############################################################################
nonlocal_val = {
    'img_interp': None,
    'text_interp': None,
    'is_multimodal': None
}

###############################################################################
# 5. Display HTML widgets for the current feature
###############################################################################

image_html_widget = widgets.HTML(value="")
text_html_widget  = widgets.HTML(value="")



# Label to show "Feature X of Y"
progress_label = widgets.Label(value="")

def load_feature(feature_index):
    """
    Read the HTML from the paths in `rnd[feature_index]` 
    and update the HTML widgets accordingly.
    """
    if feature_index < 0 or feature_index >= len(rnd):
        return
    
    feat_name, img_path, txt_path = rnd[feature_index]
    
    # Load image HTML
    try:
        with open(img_path, 'r', encoding='utf-8') as f:
            html_img = f.read().replace('margin-top: 20px;', 'margin-top: 0px;')
    except FileNotFoundError:
        html_img = f"<p style='color:red;'>Cannot find file {img_path}.</p>"
    image_html_widget.value = html_img
    
    # Load text HTML
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            html_txt = f.read().replace('margin-top: 20px;', 'margin-top: 0px;')
    except FileNotFoundError:
        html_txt = f"<p style='color:red;'>Cannot find file {txt_path}.</p>"
    text_html_widget.value = html_txt
    
    # Update labels
    progress_label.value = f"Feature {feature_index+1} of {len(rnd)}"

###############################################################################
# 6. Save & Next Feature logic
###############################################################################

feature_index = 0  # Start at the first feature

# The user will input their UID on the new Page 1
uid_input = widgets.Text(
    value='',
    description='User UID:',
    layout=widgets.Layout(width='300px')
)

save_button = widgets.Button(description='Save & Next Feature', button_style='success')
reset_button = widgets.Button(description='Reset Current', button_style='warning')

def save_and_next_feature(_):
    global feature_index
    
    if feature_index < 0 or feature_index >= len(rnd):
        return
    
    feat_name, _, _ = rnd[feature_index]
    
    # Prepare dictionary for JSON
    answers = {
        "user_uid": uid_input.value,  # read from the page 1 text box
        "feature_name": feat_name,
        "text_interp": nonlocal_val['text_interp'],
        "text_description": txt_explanation_text.value,
        "image_description": img_explanation_text.value,
        "is_multimodal": nonlocal_val['is_multimodal']
    }
    
    # Write JSON to file
    json_filename = f"annotations/user_{uid_input.value}_feature_{feat_name}_description.json"
    with open(json_filename, 'w', encoding='utf-8') as out_f:
        json.dump(answers, out_f, indent=2)
    
    # Move to the next feature
    feature_index += 1
    
    if feature_index < len(rnd):
        reset_current(None)
        load_feature(feature_index)
    else:
        reset_current(None)

def reset_current(_):
    nonlocal_val['img_interp'] = None
    nonlocal_val['text_interp'] = None
    nonlocal_val['is_multimodal'] = None
    
    reset_buttons_img()
    reset_buttons_txt()
    reset_buttons_mm()
    
    img_explanation_text.value = ''
    img_explanation_text.disabled = True
    
    txt_explanation_text.value = ''
    txt_explanation_text.disabled = True

save_button.on_click(save_and_next_feature)
reset_button.on_click(reset_current)

###############################################################################
# 7. Construct each Page
###############################################################################

# -- NEW PAGE 1: Global Instructions & UID input --
instructions_html = """
<h1>Feature Description Evaluation</h1>

<p>
Thanks so much for participating! I know y'all are busy, so it's great 
that you took the time to help me out.
</p>

<h2>Instructions</h2>

<p>
You will be presented with various features (both SAE and neurons) as well 
as samples which map out their activations in image <strong>and</strong> text space 
across four quartiles. You will receive <strong>5 images</strong> and <strong>3 text snippets</strong> drawn 
from each quartile, with the locations of feature activation highlighted.
</p>

<p>
You will be asked to determine if there exists a 
<strong>short, simple, monosemantic explanation for the feature activation</strong>. 
This boils down to finding a description that matches the types of images 
(page 2) and text samples (page 4) that the feature activates on, as well 
as where it activates, across the quartiles. Pay more attention to the samples themselves; 
we expect the activation maps to often be uninterpretable. Don't worry if your explanation 
doesn't fully match the bottom quartiles; we expect these to be usually 
uninterpretable as well.
</p>

<p>
If you cannot find a description, select <strong>no</strong>. Otherwise, select <strong>yes</strong>, 
and provide the explanation. Be sure to take your time and provide a detailed label.
</p>

<p>
After completing the answers for both text (page 5) and images (page 3), click on 
<strong>page 6</strong> to finish. If you were able to provide both text and image descriptions 
for the feature, and the descriptions approximately mean the same thing, select 
<strong>yes</strong>. Otherwise, click <strong>no</strong>.
</p>

<p>
When you are done, be sure to click <strong>Save & Next Feature</strong> to move on! 
<span style="color:red;">warning: if you do not click save, your work will be lost</span>. 
If you want, you can also reset the current feature if you need to.
</p>

<h3>Expected Experiment Length</h3>
<p>
You will be allocated 50 features to evaluate. This will take around 1-2 hours, and 
we recommend you do so in one sitting. If you leave and come back, make sure to use 
the same uid, so we know it's you. As long as you click save on each feature, your 
work will be saved (don't worry if the counter resets). Also, feel free to do less 
if you don't have time, or more features if you do! Just reload the page to get a new batch.
</p>

<h3>Incentives</h3>
<p>
bla bla bla we will pay you for your time, discussed in person!
</p>
"""

page1_instructions = widgets.VBox([
    widgets.HTML(value=instructions_html),
    widgets.HTML("<hr>"),
    widgets.HTML("<b>Please enter your UID before proceeding:</b>"),
    uid_input
])

# -- PAGE 2: Display image HTML --
page2 = widgets.VBox([
    progress_label,
    widgets.HTML("<h3>Page 2: Image HTML</h3>"),
    image_html_widget
])

# -- PAGE 3: Ask yes/no about the image explanation + optional text --
page3 = widgets.VBox([
    progress_label,
    widgets.HTML("<h3>Page 3: Explanation for Image Features?</h3>"),
    widgets.HTML("Is there a short explanation for this feature's activation over all images?"),
    img_yes_no_box,
    img_explanation_text
])

# -- PAGE 4: Display text HTML --
page4 = widgets.VBox([
    progress_label,
    widgets.HTML("<h3>Page 4: Text HTML</h3>"),
    text_html_widget
])

# -- PAGE 5: Ask yes/no about the text explanation + optional text --
page5 = widgets.VBox([
    progress_label,
    widgets.HTML("<h3>Page 5: Explanation for Text Features?</h3>"),
    widgets.HTML("Is there a short explanation for this feature's activation over all text samples?"),
    txt_yes_no_box,
    txt_explanation_text
])

# -- PAGE 6: "Do they match?" + Save & Reset --
page6 = widgets.VBox([
    progress_label,
    widgets.HTML("<h3>Page 6: Compare & Save</h3>"),
    widgets.HTML("Do your previous descriptions for text and images match?"),
    mm_yes_no_box,
    widgets.HTML("<hr>"),
    widgets.HTML("When ready, click to save answers and move on to the next feature."),
    widgets.HBox([save_button, reset_button])
])

# Assemble all pages in a Tab widget
tabs = widgets.Tab(children=[page1_instructions, page2, page3, page4, page5, page6])
titles = [
    "Page 1: Instructions",
    "Page 2: Image HTML",
    "Page 3: Img Q&A",
    "Page 4: Text HTML",
    "Page 5: Txt Q&A",
    "Page 6: Finish"
]
for i, t in enumerate(titles):
    tabs.set_title(i, t)

###############################################################################
# 8. Display the interface and load the first feature
###############################################################################
display(tabs)

if rnd:
    load_feature(feature_index)
else:
    
