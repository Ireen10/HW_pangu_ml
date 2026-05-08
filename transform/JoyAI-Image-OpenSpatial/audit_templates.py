#!/usr/bin/env python3
"""
Comprehensive template-coverage audit for infer_subtask_from_row.

For every OpenSpatial question template, instantiate it with realistic dummy
values, feed it through the classifier, and report any mismatches.

Run from the HW_pangu_ml root:
    python transform/JoyAI-Image-OpenSpatial/audit_templates.py
"""

import sys, re, textwrap
sys.path.insert(0, "transform/JoyAI-Image-OpenSpatial")
from convert_to_pangu_ml import infer_subtask_from_row


# ── helpers ──────────────────────────────────────────────────────────────────

def make_row(question: str, n_images: int = 1):
    return {
        "conversations": [{"from": "human", "value": question}],
        "images": [{"bytes": b"x"}] * n_images,
    }


def check(label: str, question: str, expected_prefix: str, n_images: int = 1):
    """Return (pass:bool, got:str)."""
    got = infer_subtask_from_row(make_row(question, n_images))
    ok = got.startswith(expected_prefix)
    return ok, got


def run_group(group_label: str, cases):
    """
    cases: list of (template_label, question_str, expected_prefix, n_images)
    Returns (n_pass, n_fail, failures)
    """
    n_pass = n_fail = 0
    failures = []
    for (lbl, q, exp, nim) in cases:
        ok, got = check(lbl, q, exp, nim)
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((lbl, got, q))
    status = "OK" if n_fail == 0 else "FAIL"
    print(f"  [{status}] {group_label}: {n_pass}/{n_pass+n_fail} passed")
    for lbl, got, q in failures:
        short_q = (q[:90] + "…") if len(q) > 90 else q
        print(f"         FAIL  {lbl!r}: got={got!r}")
        print(f"               q={short_q!r}")
    return n_pass, n_fail, failures


# ── dummy substitution values ─────────────────────────────────────────────────

OBJ_A   = "chair-(red point)"
OBJ_B   = "table-(blue point)"
OBJ_C   = "sofa-(green point)"
OBJ_LIST = "chair-(red point), table-(blue point), sofa-(green point)"
TYPE    = "objects:"
DIR     = "north"
DIST_OPT = "\nOptions: A:1.2m B:2.4m C:3.6m D:4.8m"
MCQ_OPT  = "\nOptions: A:chair-(green point) B:chair-(red point) C:chair-(pink point) D:chair-(yellow point)"
MCQ_OPT_INLINE = "A:chair-(green point) B:chair-(red point) C:chair-(pink point) D:chair-(yellow point)"
UNIT_CM  = "Calculations are in centimeters."
UNIT_M   = "Calculations are in meters."
CAM_PREAMBLE = (
    "Here are the detailed camera parameters for the image. "
    "Camera intrinsic parameters: Horizontal fov, hfov=90, and vertical fov, vfov=60. "
    "Image width=640 and height=480. We do not consider distortion parameters here. "
    "Camera coordinate: X-axis points rightward, Y-axis points downward, and Z-axis points forward. "
    "The origin point is the camera location. "
    "3D bounding box format: [x_center, y_center, z_center, x_size, y_size, z_size, pitch, yaw, roll] "
    "Output a json list where each entry contains the object name in \"label\" and its 3D bounding box in \"bbox_3d\"."
)

def fill(tpl: str, **kw) -> str:
    """Simple placeholder filler: replaces [KEY] with kw['KEY']."""
    for k, v in kw.items():
        tpl = tpl.replace(f"[{k}]", v)
    return tpl


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: correspondence  (multiview)
# ─────────────────────────────────────────────────────────────────────────────

def build_correspondence():
    cases = []
    # point2point — ABCD labels
    p2p_ABCD = [
        ('p2p[0]', 'The first image shows a point marked in red color. After adjusting the camera or lighting, the second image presents several blue points labeled "A, B, C, D". Which matches the original?'),
        ('p2p[1]', 'In image one, a point is highlighted in red color. In the second image, there are multiple blue points labeled "A, B, C, D". Can you identify the corresponding point?'),
        ('p2p[2]', 'The first image marks a point in red color. After adjusting the camera or lighting, the second image presents several blue points labeled "A, B, C, D". Which one matches the original?'),
        ('p2p[3]', 'The first image shows a point highlighted in red color. After making adjustments to the camera or lighting, the second image reveals several blue points labeled "A, B, C, D". Which point matches the original?'),
        ('p2p[4]', 'The first image features a point indicated in red color. Following adjustments to the camera or lighting, multiple blue points labeled "A, B, C, D" appear in the second image. Which one matches the original?'),
        ('p2p[5]', 'In image one, a point is indicated in red color. In the second image, there are several blue points labeled "A, B, C, D". Can you identify the corresponding point?'),
    ]
    # point2point — 1234 labels
    p2p_1234 = [
        ('p2p_num[0]', 'The first image shows a point marked in red color. After adjusting the camera or lighting, the second image presents several blue points labeled "1, 2, 3, 4". Which matches the original?'),
        ('p2p_num[1]', 'In image one, a point is highlighted in red color. In the second image, there are multiple blue points labeled "1, 2, 3, 4". Can you identify the corresponding point?'),
        ('p2p_num[2]', 'The first image marks a point in red color. After adjusting the camera or lighting, the second image presents several blue points labeled "1, 2, 3, 4". Which one matches the original?'),
        ('p2p_num[3]', 'The first image shows a point highlighted in red color. After making adjustments to the camera or lighting, the second image reveals several blue points labeled "1, 2, 3, 4". Which point matches the original?'),
        ('p2p_num[4]', 'The first image features a point indicated in red color. Following adjustments to the camera or lighting, multiple blue points labeled "1, 2, 3, 4" appear in the second image. Which one matches the original?'),
        ('p2p_num[5]', 'In image one, a point is indicated in red color. In the second image, there are several blue points labeled "1, 2, 3, 4". Can you identify the corresponding point?'),
    ]
    # MCQ variant (task appends " Options: A point-A B point-B C point-C D point-D")
    p2p_mcq = [
        ('p2p_mcq[0]', 'The first image shows a point marked in red color. After adjusting the camera or lighting, the second image presents several blue points labeled "A, B, C, D". Which matches the original? Options: A point-A B point-B C point-C D point-D'),
        ('p2p_mcq[1]', 'The first image features a point indicated in red color. Following adjustments to the camera or lighting, multiple blue points labeled "1, 2, 3, 4" appear in the second image. Which one matches the original? Options: A point-1 B point-2 C point-3 D point-4'),
    ]
    # object2object
    o2o = [
        ('o2o[0]', 'Does the chair in image 1 show up in image 2?'),
        ('o2o[1]', 'Can you find the chair from image 1 in image 2?'),
        ('o2o[2]', 'Is the chair from the first image visible in the second image?'),
        ('o2o[3]', 'Is the chair in image 1 different from any object in image 2?'),
    ]
    for lbl, q in p2p_ABCD + p2p_1234 + p2p_mcq + o2o:
        cases.append((lbl, q, "correspondence.multi_view", 2))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: position — multiview (type1 + type2, open-ended + MCQ "\n[O]" appended)
# ─────────────────────────────────────────────────────────────────────────────

def build_position_multiview():
    type1_templates = [
        ("If the [A] is [X] of the [B] in image 1, what direction is the [C] (visible in image 2) from the [B]?", "pos_mv_t1[0]"),
        ("If the [A] is to the [X] of the [B] in the first image, what direction is the [C] from the [B]?", "pos_mv_t1[1]"),
        ("Given that the [A] appears [X] relative to the [B] in image 1, which direction does the [C] (seen in image 2) lie with respect to the [B]?", "pos_mv_t1[2]"),
        ("In image 1, if the [A] is located [X] of the [B], what direction does the [C] (depicted in image 2) take from the [B]?", "pos_mv_t1[3]"),
        ("If the [A] is positioned [X] relative to the [B] in the first image, how would you describe the direction of the [C] (visible in image 2) in relation to the [B]?", "pos_mv_t1[4]"),
        ("What direction does the [C] (shown in image 2) occupy from the [B], given that the [A] is [X] to the [B] in image 1?", "pos_mv_t1[5]"),
    ]
    type2_templates = [
        ("If I am at the position of the [B] in image 1, and the [A] is on the [X] side of me, what direction is the [C] (visible in image 2) from my position?", "pos_mv_t2[0]"),
        ("Standing at the location of the [B] in the first image, with the [A] is on my [X] side, which direction does the [C] (seen in image 2 ) lie from me?", "pos_mv_t2[1]"),
        ("From the viewpoint of the [B] in image 1, if the [A] is located at the [X] side of me, what direction does the [C] (depicted in image 2) take from my position?", "pos_mv_t2[2]"),
        ("If I consider myself at the [B]'s position in the first image, and the [A] is positioned at the [X] side of me, how would I describe the direction of the [C] (visible in image 2) from my location?", "pos_mv_t2[3]"),
        ("Assume I am at the [B]'s position in image 1, with the [A] on my [X] side, what direction does the [C] (shown in image 2) occupy from my viewpoint?", "pos_mv_t2[4]"),
        ("From the perspective of the [B] in the first image, if the [A] is on the [X] side of the [B], which direction is the [C] (visible in image 2) from the [B]'s position?", "pos_mv_t2[5]"),
    ]
    cases = []
    mcq_opts = "\nOptions: A:north B:south C:east D:west"
    for tpl, lbl in type1_templates + type2_templates:
        q_oe = fill(tpl, A=OBJ_A, B=OBJ_B, C=OBJ_C, X="left")
        q_mcq = fill(tpl, A=OBJ_A, B=OBJ_B, C=OBJ_C, X="left") + mcq_opts
        cases.append((lbl + "_oe",  q_oe,  "position.multi_view", 2))
        cases.append((lbl + "_mcq", q_mcq, "position.multi_view", 2))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: position — single_view (height_higher, height_lower, next_far)
# ─────────────────────────────────────────────────────────────────────────────

def build_position_singleview():
    higher_templates = [
        ("Consider the real-world 3D locations of the objects. Which object has a higher location? [O]", "pos_sv_higher[0]"),
        ("Based on the 3D positions of the objects, which one is placed at a higher elevation? [O]", "pos_sv_higher[1]"),
        ("Looking at the real-world 3D arrangement, which object is positioned higher? [O]", "pos_sv_higher[2]"),
        ("Considering the spatial positions of the objects in 3D space, which one sits higher? [O]", "pos_sv_higher[3]"),
    ]
    lower_templates = [
        ("Consider the real-world 3D locations of the objects. Which object has a lower location? [O]", "pos_sv_lower[0]"),
        ("Based on the 3D positions of the objects, which one is placed at a lower elevation? [O]", "pos_sv_lower[1]"),
        ("Looking at the real-world 3D arrangement, which object is positioned lower? [O]", "pos_sv_lower[2]"),
        ("Considering the spatial positions of the objects in 3D space, which one sits lower? [O]", "pos_sv_lower[3]"),
    ]
    next_far_templates = [
        ("Consider the real-world 3D locations of the objects. Are the [A] and the [B] next to each other or far away from each other? [O]", "pos_sv_nextfar[0]"),
        ("Based on the 3D spatial arrangement, are the [A] and the [B] close together or far apart? [O]", "pos_sv_nextfar[1]"),
        ("Looking at the real-world positions of the objects, are the [A] and the [B] near each other or distant? [O]", "pos_sv_nextfar[2]"),
        ("Considering the spatial layout, would you say the [A] and the [B] are adjacent or separated by a large distance? [O]", "pos_sv_nextfar[3]"),
    ]
    cases = []
    # OE: [O] = "The chair or the table?"
    oe_o = f"The {OBJ_A} or the {OBJ_B}?"
    # MCQ: [O] = "\nOptions: A:The chair B:The table"
    mcq_o = f"\nOptions: A:The {OBJ_A} B:The {OBJ_B}"
    # MCQ proximity: [O] = "\nOptions: A: next to each other B: far away from each other"
    mcq_o_prox = "\nOptions: A: next to each other B: far away from each other"

    for tpl, lbl in higher_templates + lower_templates:
        cases.append((lbl + "_oe",  fill(tpl, O=oe_o),   "position.single_view", 1))
        cases.append((lbl + "_mcq", fill(tpl, O=mcq_o),  "position.single_view", 1))
    for tpl, lbl in next_far_templates:
        q_oe  = fill(tpl, A=OBJ_A, B=OBJ_B, O=oe_o)
        q_mcq = fill(tpl, A=OBJ_A, B=OBJ_B, O=mcq_o_prox)
        cases.append((lbl + "_oe",  q_oe,  "position.single_view", 1))
        cases.append((lbl + "_mcq", q_mcq, "position.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: depth  (single_view)
# ─────────────────────────────────────────────────────────────────────────────

def build_depth():
    ordering_oe = [
        ("depth_ord[0]", f"Given the objects: {OBJ_LIST}, please order them by depth (from near to far)."),
        ("depth_ord[1]", f"Please arrange the objects: {OBJ_LIST} based on their depth (from near to far)."),
        ("depth_ord[2]", f"Order the objects: {OBJ_LIST} according to their depth from near to far."),
        ("depth_ord[3]", f"Sort the objects: {OBJ_LIST} by depth (from near to far)."),
        ("depth_ord[4]", f"Can you organize the objects: {OBJ_LIST} in order of their depth (from near to far)?"),
        ("depth_ord[5]", f"Please sequence the objects: {OBJ_LIST} from shallowest to deepest ."),
    ]
    ordering_mcq = [
        ("depth_ord_mcq[0]", f"Given the objects: {OBJ_LIST}, please order them by depth (from near to far). Consider the following options: {MCQ_OPT_INLINE} and choose the correct one."),
        ("depth_ord_mcq[1]", f"Please arrange the objects: {OBJ_LIST} based on their depth (from near to far). Please consider the following options: {MCQ_OPT_INLINE}, and choose the correct one."),
        ("depth_ord_mcq[2]", f"Order the objects: {OBJ_LIST} according to their depth from near to far. Think about these options: {MCQ_OPT_INLINE}. Which one do you believe is correct?"),
        ("depth_ord_mcq[3]", f"Sort the objects: {OBJ_LIST} by depth (from near to far). Here are the options to choose from: {MCQ_OPT_INLINE}. Please select the correct answer."),
        ("depth_ord_mcq[4]", f"Can you organize the objects: {OBJ_LIST} in order of their depth (from near to far)? Consider these options: {MCQ_OPT_INLINE}, and choose the correct answer."),
        ("depth_ord_mcq[5]", f"Please sequence the objects: {OBJ_LIST} from shallowest to deepest . Before making a decision, please review the following options: {MCQ_OPT_INLINE}, and select the correct one."),
    ]
    choice_oe = [
        ("depth_choice[0]", f"Between the 3 objects {OBJ_LIST}, which one is the 2nd closest to the camera?"),
        ("depth_choice[1]", f"Among the 3 objects {OBJ_LIST}, which one is the 2nd nearest to the camera?"),
        ("depth_choice[2]", f"From the 3 objects {OBJ_LIST}, identify the one that is the 2nd closest to the camera."),
        ("depth_choice[3]", f"Considering the 3 objects {OBJ_LIST}, which one is the 2nd nearest to the camera?"),
        ("depth_choice[4]", f"Out of the 3 objects {OBJ_LIST}, which one has the 2nd smallest depth?"),
    ]
    choice_mcq = [
        ("depth_choice_mcq[0]", f"Between the objects {OBJ_LIST}, which one is the 2nd closest to the camera? Consider the following options: {MCQ_OPT_INLINE} and choose the correct one."),
        ("depth_choice_mcq[1]", f"Among the objects {OBJ_LIST}, which one is the 2nd nearest to the camera? Please consider the following options: {MCQ_OPT_INLINE}, and choose the correct one."),
        ("depth_choice_mcq[2]", f"From the objects {OBJ_LIST}, identify the one that is the 2nd closest to the camera. Think about these options: {MCQ_OPT_INLINE}. Which one do you believe is correct?"),
        ("depth_choice_mcq[3]", f"Considering the objects {OBJ_LIST}, which one is the 2nd nearest to the camera? Here are the options to choose from: {MCQ_OPT_INLINE}. Please select the correct answer."),
        ("depth_choice_mcq[4]", f"Out of the objects {OBJ_LIST}, which one has the 2nd smallest depth? Consider these options: {MCQ_OPT_INLINE}, and choose the correct answer."),
    ]
    farthest_oe = [
        ("depth_far[0]", f"Between the 3 objects {OBJ_LIST}, which one is the farthest from the camera?"),
        ("depth_far[1]", f"Among the 3 objects {OBJ_LIST}, which one is the most distant from the camera?"),
        ("depth_far[2]", f"From the 3 objects {OBJ_LIST}, identify the one that is the farthest from the camera."),
        ("depth_far[3]", f"Considering the 3 objects {OBJ_LIST}, which one is the most distant from the camera?"),
        ("depth_far[4]", f"Out of the 3 objects {OBJ_LIST}, which one has the greatest depth?"),
        ("depth_far[5]", f"From the 3 objects {OBJ_LIST}, which is the one with the largest depth?"),
    ]
    farthest_mcq = [
        ("depth_far_mcq[0]", f"Between the objects {OBJ_LIST}, which one is the farthest from the camera? Consider the following options: {MCQ_OPT_INLINE} and choose the correct one."),
        ("depth_far_mcq[1]", f"Among the objects {OBJ_LIST}, which one is the most distant from the camera? Please consider the following options: {MCQ_OPT_INLINE}, and choose the correct one."),
        ("depth_far_mcq[2]", f"From the objects {OBJ_LIST}, identify the one that is the farthest from the camera. Think about these options: {MCQ_OPT_INLINE}. Which one do you believe is correct?"),
        ("depth_far_mcq[3]", f"Considering the objects {OBJ_LIST}, which one is the most distant from the camera? Here are the options to choose from: {MCQ_OPT_INLINE}. Please select the correct answer."),
        ("depth_far_mcq[4]", f"Out of the objects {OBJ_LIST}, which one has the greatest depth? Consider these options: {MCQ_OPT_INLINE}, and choose the correct answer."),
        # MCQ[5] uses "which one is" not "which is" — wording inconsistency in OpenSpatial source
        ("depth_far_mcq[5]", f"From the objects {OBJ_LIST}, which one is the one with the largest depth? Before making a decision, please review the following options: {MCQ_OPT_INLINE}, and select the correct one."),
    ]
    closest_oe = [
        ("depth_close[0]", f"Between the 3 objects {OBJ_LIST}, which one is the closest to the camera?"),
        ("depth_close[1]", f"Among the 3 objects {OBJ_LIST}, which one is the nearest to the camera?"),
        ("depth_close[2]", f"From the 3 objects {OBJ_LIST}, identify the one that is the closest to the camera."),
        ("depth_close[3]", f"Considering the 3 objects {OBJ_LIST}, which one is the nearest to the camera?"),
        ("depth_close[4]", f"Out of the 3 objects {OBJ_LIST}, which one has the smallest depth?"),
        ("depth_close[5]", f"From the 3 objects {OBJ_LIST}, which one is the one with the least depth?"),
    ]
    closest_mcq = [
        ("depth_close_mcq[0]", f"Between the objects {OBJ_LIST}, which one is the closest to the camera? Consider the following options: {MCQ_OPT_INLINE} and choose the correct one."),
        ("depth_close_mcq[1]", f"Among the objects {OBJ_LIST}, which one is the nearest to the camera? Please consider the following options: {MCQ_OPT_INLINE}, and choose the correct one."),
        ("depth_close_mcq[2]", f"From the objects {OBJ_LIST}, identify the one that is the closest to the camera. Think about these options: {MCQ_OPT_INLINE}. Which one do you believe is correct?"),
        ("depth_close_mcq[3]", f"Considering the objects {OBJ_LIST}, which one is the nearest to the camera? Here are the options to choose from: {MCQ_OPT_INLINE}. Please select the correct answer."),
        ("depth_close_mcq[4]", f"Out of the objects {OBJ_LIST}, which one has the smallest depth? Consider these options: {MCQ_OPT_INLINE}, and choose the correct answer."),
        ("depth_close_mcq[5]", f"From the objects {OBJ_LIST}, which one is the one with the least depth? Before making a decision, please review the following options: {MCQ_OPT_INLINE}, and select the correct one."),
    ]
    cases = []
    for lbl, q in (ordering_oe + ordering_mcq + choice_oe + choice_mcq +
                    farthest_oe + farthest_mcq + closest_oe + closest_mcq):
        cases.append((lbl, q, "depth.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: counting  (single_view)
# ─────────────────────────────────────────────────────────────────────────────

def build_counting():
    oe = [
        ("cnt[0]", "Find out how many chairs in this scene."),
        ("cnt[1]", "What is the number of the chairs?"),
        ("cnt[2]", "How many chairs are there?"),
        ("cnt[3]", "Could you tell me the number of the chairs?"),
        ("cnt[4]", "Counting the number of chairs in this scene?"),
        ("cnt[5]", "How many chairs can you see?"),
        ("cnt[6]", "How many chairs are present?"),
        ("cnt[7]", "What is the count of the chairs?"),
        ("cnt[8]", "Can you provide the count of the chair?"),
        ("cnt[9]", "Please count the number of chair."),
    ]
    mcq = [
        ("cnt_mcq[0]", "Find out how many chairs in this scene. Please consider the following options: A:1 B:2 C:3 D:4, and choose the correct one."),
        ("cnt_mcq[1]", "What is the number of the chairs? Think about these options: A:1 B:2 C:3 D:4. Which one do you believe is correct?"),
        ("cnt_mcq[2]", "How many chairs are there? Refer to the following options: A:1 B:2 C:3 D:4, and pick the one you think is correct."),
        ("cnt_mcq[3]", "Could you tell me the number of the chairs? Take a moment to carefully consider the following options: A:1 B:2 C:3 D:4, and choose the correct one."),
        ("cnt_mcq[4]", "How many chairs can you see? Here are the options to choose from: A:1 B:2 C:3 D:4. Please select the correct answer."),
        ("cnt_mcq[5]", "How many chairs are present? Here are the options to choose from: A:1 B:2 C:3 D:4. Please select the correct answer."),
        ("cnt_mcq[6]", "What is the count of the chairs? Consider the following options: A:1 B:2 C:3 D:4 and choose the correct one."),
        ("cnt_mcq[7]", "Can you provide the count of the chair? Consider the following options: A:1 B:2 C:3 D:4 and choose the correct one."),
        ("cnt_mcq[8]", "Please count the number of chair. Consider the following options: A:1 B:2 C:3 D:4 and choose the correct one."),
    ]
    cases = []
    for lbl, q in oe + mcq:
        cases.append((lbl, q, "counting.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: size — multiview (biggest, smallest, big, small)
# ─────────────────────────────────────────────────────────────────────────────

def build_size_multiview():
    biggest = [
        ("sz_mv_big[0]", f"Given the multi-view images and the objects: {OBJ_LIST}, which one is the biggest?"),
        ("sz_mv_big[1]", f"Considering the set of objects: {OBJ_LIST} in the multi-view images, identify the one with the largest size."),
        ("sz_mv_big[2]", f"From the provided objects: {OBJ_LIST} in different perspectives, which object has the greatest size?"),
        ("sz_mv_big[3]", f"Out of the objects: {OBJ_LIST}, which one is the largest in size?"),
        ("sz_mv_big[4]", f"From the collection of objects: {OBJ_LIST} in different views, determine which is the biggest."),
    ]
    smallest = [
        ("sz_mv_small[0]", f"Given the multi-view images and the objects: {OBJ_LIST}, which one is the smallest?"),
        ("sz_mv_small[1]", f"Considering the set of objects: {OBJ_LIST} in the multi-view images, identify the one with the smallest size."),
        ("sz_mv_small[2]", f"From the provided objects: {OBJ_LIST} in different perspectives, which object has the least size?"),
        ("sz_mv_small[3]", f"Out of the objects: {OBJ_LIST}, which one is the smallest in size?"),
        ("sz_mv_small[4]", f"From the collection of objects: {OBJ_LIST} in different views, determine which is the smallest."),
    ]
    big = [
        ("sz_mv_bigger[0]", f"Given two different views, Is the {OBJ_A} bigger than the {OBJ_B}?"),
        ("sz_mv_bigger[1]", f"As shown in different views, does the {OBJ_A} have a larger size compared to the {OBJ_B}?"),
        ("sz_mv_bigger[2]", f"After reviewing the images, can you confirm if the {OBJ_A} is bigger than the {OBJ_B}?"),
    ]
    small = [
        ("sz_mv_smaller[0]", f"Based on the given images, is the {OBJ_A} smaller than the {OBJ_B}?"),
        ("sz_mv_smaller[1]", f"Considering the different perspectives of the scene, does the {OBJ_A} have a smaller size compared to the {OBJ_B}?"),
        ("sz_mv_smaller[2]", f"After reviewing the images, can you confirm if the {OBJ_A} is smaller than the {OBJ_B}?"),
    ]
    cases = []
    for lbl, q in biggest + smallest + big + small:
        cases.append((lbl, q, "size.multi_view", 2))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: size — single_view (absolute, height, big, small)
# ─────────────────────────────────────────────────────────────────────────────

def build_size_singleview():
    absolute = [
        ("sz_sv_abs[0]", f"What is the length of the dimension that is largest in size (length, width, or height) of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_abs[1]", f"What is the measurement for the longest side (length, width, or height) of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_abs[2]", f"Can you provide the size of the {OBJ_A}'s largest dimension (length, width, or height)? {UNIT_CM}"),
        ("sz_sv_abs[3]", f"What is the length of the dimension that is maximum (length, width, or height) of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_abs[4]", f"What is the length of the dimension that is the greatest (length, width, or height) of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_abs[5]", f"What is the measurement of the {OBJ_A}'s longest dimension (length, width, or height)? {UNIT_M}"),
        ("sz_sv_abs[6]", f"Can you tell me the size of the {OBJ_A}'s maximum dimension (length, width, or height)? {UNIT_M}"),
        ("sz_sv_abs[7]", f"What is the length of the dimension that is the most extensive (length, width, or height) of the {OBJ_A}? {UNIT_M}"),
        ("sz_sv_abs[8]", f"What is the measurement of the {OBJ_A}'s greatest dimension (length, width, or height)? {UNIT_M}"),
        ("sz_sv_abs[9]", f"Can you provide the size of the {OBJ_A}'s most significant dimension (length, width, or height)? {UNIT_M}"),
    ]
    height = [
        ("sz_sv_h[0]", f"Could you estimate the height of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_h[1]", f"What is the vertical measurement of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_h[2]", f"Can you provide the height dimension of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_h[3]", f"How tall does the {OBJ_A} stand? {UNIT_CM}"),
        ("sz_sv_h[4]", f"What is the height of the {OBJ_A}? {UNIT_CM}"),
        ("sz_sv_h[5]", f"Could you tell me the vertical size of the {OBJ_A}? {UNIT_M}"),
        ("sz_sv_h[6]", f"What is the measurement of the {OBJ_A}'s height? {UNIT_M}"),
        ("sz_sv_h[7]", f"Can you estimate how high the {OBJ_A} is? {UNIT_M}"),
        ("sz_sv_h[8]", f"What is the vertical dimension of the {OBJ_A}? {UNIT_M}"),
    ]
    big = [
        ("sz_sv_big[0]", f"Is the {OBJ_A} bigger than the {OBJ_B}?"),
        ("sz_sv_big[1]", f"Does the {OBJ_A} have a larger size compared to the {OBJ_B}?"),
        ("sz_sv_big[2]", f"Can you confirm if the {OBJ_A} is bigger than the {OBJ_B}?"),
    ]
    small = [
        ("sz_sv_small[0]", f"Is the {OBJ_A} smaller than the {OBJ_B}?"),
        ("sz_sv_small[1]", f"Does the {OBJ_A} have a smaller size compared to the {OBJ_B}?"),
        ("sz_sv_small[2]", f"Can you confirm if the {OBJ_A} is smaller than the {OBJ_B}?"),
    ]
    cases = []
    for lbl, q in absolute + height + big + small:
        cases.append((lbl, q, "size.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: distance — multiview (farthest, closest, obj_cam, obj_cam_mcq)
# ─────────────────────────────────────────────────────────────────────────────

def build_distance_multiview():
    farthest = [
        ("dist_mv_far[0]", f"Given the multi-view images and objects: {OBJ_LIST}, which one is the farthest from the camera?"),
        ("dist_mv_far[1]", f"Considering the multi-view images and the set of objects {OBJ_LIST}, which object is most distant from camera?"),
        ("dist_mv_far[2]", f"From the provided multi-view images and objects {OBJ_LIST}, identify the object that is the farthest from camera."),
        ("dist_mv_far[3]", f"Among the objects {OBJ_LIST} shown in the multi-view images, which one has the greatest distance from camera?"),
        ("dist_mv_far[4]", f"From the multi-view objects {OBJ_LIST}, identify the one farthest from camera."),
        ("dist_mv_far[5]", f"Out of the objects {OBJ_LIST} in the multi-view images, which one is the most distant from camera?"),
        ("dist_mv_far[6]", f"If you view objects {OBJ_LIST} from multiple perspectives, which one has the maximum distance to camera?"),
    ]
    closest = [
        ("dist_mv_close[0]", f"Given the multi-view images and objects: {OBJ_LIST}, which one is the closest to the camera?"),
        ("dist_mv_close[1]", f"Considering the multi-view images and the set of objects {OBJ_LIST}, which object is nearest to camera?"),
        ("dist_mv_close[2]", f"From the provided multi-view images and objects {OBJ_LIST}, identify the object that is the closest to camera."),
        ("dist_mv_close[3]", f"Among the objects {OBJ_LIST} shown in the multi-view images, which one has the smallest distance from camera?"),
        ("dist_mv_close[4]", f"From the multi-view objects {OBJ_LIST}, identify the one closest to camera."),
        ("dist_mv_close[5]", f"Out of the objects {OBJ_LIST} in the multi-view images, which one is the nearest to camera?"),
        ("dist_mv_close[6]", f"If you view objects {OBJ_LIST} from multiple perspectives, which one has the minimum distance to camera?"),
    ]
    obj_cam_oe = [
        ("dist_mv_cam[0]", f"View 1 and View 2 are two different views that represent the same scene. In which view the {OBJ_A} in the scene is closer to the spot where the camera view was positioned?"),
        ("dist_mv_cam[1]", f"Two views (View 1 and View 2) show the same scene from different angles. In which view is the {OBJ_A} closer to the camera position?"),
        ("dist_mv_cam[2]", f"Given View 1 and View 2 of the same scene, in which view does the {OBJ_A} appear closer to where the camera was placed?"),
        ("dist_mv_cam[3]", f"The same scene is captured in View 1 and View 2. In which view is the {OBJ_A} closer to the camera viewpoint?"),
    ]
    obj_cam_mcq = [
        ("dist_mv_cam_mcq[0]", f"View 1 and View 2 are two different views that represent the same scene. In which view the {OBJ_A} in the scene is closer to the spot where the camera view was positioned?\nOptions: A:View 1 B:View 2"),
        ("dist_mv_cam_mcq[1]", f"Two views (View 1 and View 2) show the same scene from different angles. In which view is the {OBJ_A} closer to the camera position?\nOptions: A:View 1 B:View 2"),
        ("dist_mv_cam_mcq[2]", f"Given View 1 and View 2 of the same scene, in which view does the {OBJ_A} appear closer to where the camera was placed?\nOptions: A:View 1 B:View 2"),
        ("dist_mv_cam_mcq[3]", f"The same scene is captured in View 1 and View 2. In which view is the {OBJ_A} closer to the camera viewpoint?\nOptions: A:View 1 B:View 2"),
    ]
    cases = []
    for lbl, q in farthest + closest + obj_cam_oe + obj_cam_mcq:
        cases.append((lbl, q, "distance.multi_view", 2))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: distance — neutral (img_count decides sv/mv)
# ─────────────────────────────────────────────────────────────────────────────

def build_distance_neutral():
    absolute = [
        ("dist_abs_m[0]",  f"Measuring from the closest point of each object, what is the distance between the {OBJ_A} and the {OBJ_B} (in meters)?"),
        ("dist_abs_cm[0]", f"Measuring from the closest point of each object, what is the distance between the {OBJ_A} and the {OBJ_B} (in centimeters)?"),
        ("dist_abs_m[1]",  f"What is the distance between the {OBJ_A} and the {OBJ_B} (in meters)?"),
        ("dist_abs_cm[1]", f"What is the distance between the {OBJ_A} and the {OBJ_B} (in centimeters)?"),
        ("dist_abs_m[2]",  f"Consider the real-world 3D location of the objects. What is the distance between the {OBJ_A} and the {OBJ_B} (in meters)?"),
        ("dist_abs_cm[2]", f"Consider the real-world 3D location of the objects. What is the distance between the {OBJ_A} and the {OBJ_B} (in centimeters)?"),
    ]
    rel_far = [
        ("dist_relfar[0]", f"Estimate the real-world distances between objects in this image. Which object is farther from the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relfar[1]", f"Based on the spatial arrangement of objects in this image, which object is more distant from the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relfar[2]", f"Considering the 3D positions of objects in this image, which one is farther from the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relfar[3]", f"From the perspective of this image, which object is more distant from the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relfar[4]", f"Looking at the spatial layout in this image, which object is farther from the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relfar[5]", f"Which of {OBJ_A} and {OBJ_B} is farther to {OBJ_C}? The {OBJ_A}."),
    ]
    rel_close = [
        ("dist_relclose[0]", f"Estimate the real-world distances between objects in this image. Which object is closer to the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relclose[1]", f"Based on the spatial arrangement of objects in this image, which object is nearer to the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relclose[2]", f"Considering the 3D positions of objects in this image, which one is closer to the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relclose[3]", f"From the perspective of this image, which object is nearer to the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relclose[4]", f"Looking at the spatial layout in this image, which object is closer to the {OBJ_C}, the {OBJ_A} or the {OBJ_B}? The {OBJ_A}."),
        ("dist_relclose[5]", f"Which of {OBJ_A} and {OBJ_B} is closer to {OBJ_C}? The {OBJ_A}."),
    ]
    cases = []
    for lbl, q in absolute:
        cases.append((lbl + "_sv", q, "distance.single_view", 1))
        cases.append((lbl + "_mv", q, "distance.multi_view",  2))
    for lbl, q in rel_far + rel_close:
        cases.append((lbl + "_sv", q, "distance.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: grounding_3d  (single_view; camera preamble prepended)
# ─────────────────────────────────────────────────────────────────────────────

def build_grounding_3d():
    oe_templates = [
        f"Identify the 3D bounding box surrounding the {OBJ_A} within this environment.",
        f"Locate the 3D bounding volume for the {OBJ_A} present in the scene.",
        f"Find the 3D bounding box that encapsulates the {OBJ_A} in this visual representation.",
        f"Extract the 3D bounding box coordinates of the {OBJ_A} located in the image.",
        f"Outline the 3D bounding box for the {OBJ_A} visible in this setting.",
        f"Pinpoint the 3D bounding box enclosing the {OBJ_A} in this layout.",
        f"Trace the edges of the 3D bounding box around the {OBJ_A} in this scenario.",
        f"Highlight the 3D bounding box that frames the {OBJ_A} observed in the image.",
        f"Predict the 3D location of the {OBJ_A} observed in the image.",
    ]
    mcq_templates = [
        f"Identify the 3D bounding box surrounding the {OBJ_A} within this environment. Consider the following options: A:[1,2,3] B:[4,5,6] and choose the correct one.",
        f"Locate the 3D bounding volume for the {OBJ_A} present in the scene. Please consider the following options: A:[1,2,3] B:[4,5,6], and choose the correct one.",
        f"Determine the dimensions of the 3D bounding box for the {OBJ_A} in this context. Think about these options: A:[1,2,3]. Which one do you believe is correct?",
        f"Find the 3D bounding box that encapsulates the {OBJ_A} in this visual representation. Here are the options to choose from: A:[1,2,3]. Please select the correct answer.",
        f"Extract the 3D bounding box coordinates of the {OBJ_A} located in the image. Consider these options: A:[1,2,3], and choose the correct answer.",
        f"Outline the 3D bounding box for the {OBJ_A} visible in this setting. Before making a decision, please review the following options: A:[1,2,3], and select the correct one.",
        f"Calculate the 3D bounding box dimensions for the {OBJ_A} depicted in the scene. Take a moment to carefully consider the following options: A:[1,2,3], and choose the correct one.",
        f"Pinpoint the 3D bounding box enclosing the {OBJ_A} in this layout. Refer to the following options: A:[1,2,3], and pick the one you think is correct.",
        f"Trace the edges of the 3D bounding box around the {OBJ_A} in this scenario. Refer to the following options: A:[1,2,3], and pick the one you think is correct.",
        f"Highlight the 3D bounding box that frames the {OBJ_A} observed in the image. Please consider the following options: A:[1,2,3], and choose the correct one.",
        f"Predict the 3D location of the {OBJ_A} observed in the image. Think about these options: A:[1,2,3]. Which one do you believe is correct?",
    ]
    # camera_system prompt (the system prompt itself becomes the question when used as a "camera_system" task)
    cam_system = CAM_PREAMBLE

    cases = []
    for i, q in enumerate(oe_templates):
        # With camera preamble prepended
        cases.append((f"g3d_oe[{i}]_no_preamble", q, "grounding_3d.single_view", 1))
        cases.append((f"g3d_oe[{i}]_with_preamble", CAM_PREAMBLE + " " + q, "grounding_3d.single_view", 1))
    for i, q in enumerate(mcq_templates):
        cases.append((f"g3d_mcq[{i}]_no_preamble", q, "grounding_3d.single_view", 1))
        cases.append((f"g3d_mcq[{i}]_with_preamble", CAM_PREAMBLE + " " + q, "grounding_3d.single_view", 1))
    # camera_system standalone
    cases.append(("g3d_cam_system", cam_system, "grounding_3d.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY: 3d_scene_caption  (single_view; assembled from modules)
# ─────────────────────────────────────────────────────────────────────────────

def build_caption():
    task_modules = [
        "Create comprehensive spatial relationship descriptions that capture every observable detail in 100-200 words.",
        "Generate systematic visual documentation focusing on spatial relationships of object positions in 100-200 words.",
        "Develop detailed scene inventories that catalog all visible elements and their spatial relationships in 100-200 words.",
        "Produce structured spatial layout analysis report containing both descriptive text and technical metadata in 100-200 words.",
        "Construct thorough image assessments covering spatial, temporal, and contextual elements in 100-200 words.",
    ]
    role = "You are a professional image analyst specializing in detailed visual spatial analysis."
    subject = "Focus on primary subjects including their positioning, appearance, actions, and interactions."
    constraint = "Maintain objective documentation without subjective interpretations, emotional language, or aesthetic judgments."

    cases = []
    for i, task in enumerate(task_modules):
        # Task module alone (minimum case — dropout=0 means task is always present)
        cases.append((f"caption_task_only[{i}]", task, "3d_scene_caption.single_view", 1))
        # With role prepended
        cases.append((f"caption_role+task[{i}]", role + " " + task, "3d_scene_caption.single_view", 1))
        # Full assembled prompt
        full = " ".join([role, task, subject, constraint])
        cases.append((f"caption_full[{i}]", full, "3d_scene_caption.single_view", 1))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_pass = all_fail = 0
    all_failures = []

    groups = [
        ("correspondence    (multiview)",          build_correspondence()),
        ("position          (multiview)",          build_position_multiview()),
        ("position          (single_view)",        build_position_singleview()),
        ("depth             (single_view)",        build_depth()),
        ("counting          (single_view)",        build_counting()),
        ("size              (multiview)",          build_size_multiview()),
        ("size              (single_view)",        build_size_singleview()),
        ("distance          (multiview)",          build_distance_multiview()),
        ("distance          (neutral sv+mv)",      build_distance_neutral()),
        ("grounding_3d      (single_view)",        build_grounding_3d()),
        ("3d_scene_caption  (single_view)",        build_caption()),
    ]

    print("=" * 70)
    print("OpenSpatial template coverage audit")
    print("=" * 70)
    for label, cases in groups:
        n_pass, n_fail, failures = run_group(label, cases)
        all_pass += n_pass
        all_fail += n_fail
        all_failures.extend(failures)

    print("=" * 70)
    print(f"TOTAL: {all_pass}/{all_pass + all_fail} passed,  {all_fail} failed")
    if all_fail == 0:
        print("All templates covered!")
    else:
        print(f"\n{'─'*70}")
        print("Failures summary:")
        for lbl, got, q in all_failures:
            short_q = (q[:100] + "…") if len(q) > 100 else q
            print(f"  {lbl!r}  got={got!r}")
            print(f"    q={short_q!r}")


if __name__ == "__main__":
    main()
