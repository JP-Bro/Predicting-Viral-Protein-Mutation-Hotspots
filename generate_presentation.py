"""
generate_presentation.py
------------------------
Generates a 10-slide PowerPoint presentation (.pptx) with modern dark/cyan styling,
structured cards, tables, formulas, and ELI5 explanations.

Outputs:
  reports/Viral_Mutation_Escape_Prediction.pptx
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Styling Palette
# ---------------------------------------------------------------------------
COLOR_BG_DARK     = RGBColor(15, 23, 42)     # Slate 900
COLOR_CARD_DARK   = RGBColor(30, 41, 59)     # Slate 800
COLOR_TEXT_WHITE  = RGBColor(248, 250, 252) # Slate 50
COLOR_TEXT_MUTED  = RGBColor(148, 163, 184) # Slate 400
COLOR_CYAN        = RGBColor(56, 189, 248)   # Sky 400
COLOR_BLUE        = RGBColor(99, 102, 241)   # Indigo 500
COLOR_GREEN       = RGBColor(52, 211, 153)   # Emerald 400
COLOR_AMBER       = RGBColor(251, 191, 36)   # Amber 400

FONT_NAME = "Segoe UI"


def set_slide_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_DARK, border_color=COLOR_CYAN):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def add_header(slide, title_text, subtitle_text):
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.1))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_NAME
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_NAME
        p2.font.size = Pt(14)
        p2.font.color.rgb = COLOR_TEXT_MUTED


def build_presentation(output_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -----------------------------------------------------------------------
    # SLIDE 1: Title
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)

    # Decorative Card
    add_card(slide, Inches(1.5), Inches(1.5), Inches(10.333), Inches(4.5), border_color=COLOR_CYAN)

    txBox = slide.shapes.add_textbox(Inches(2.0), Inches(2.0), Inches(9.333), Inches(3.5))
    tf = txBox.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Viral Mutation & Immune Escape Prediction"
    p.font.name = FONT_NAME
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Multimodal Deep Learning (ESM-2 + GVP-GNN + Cross-Attention)"
    p2.font.name = FONT_NAME
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLOR_TEXT_WHITE
    p2.alignment = PP_ALIGN.CENTER

    p3 = tf.add_paragraph()
    p3.text = "\nTeam: Aditya (Data) · Vineela (Architecture) · Jimit (Evaluation)\nSARS-CoV-2 Spike Protein Surveillance"
    p3.font.name = FONT_NAME
    p3.font.size = Pt(14)
    p3.font.color.rgb = COLOR_TEXT_MUTED
    p3.alignment = PP_ALIGN.CENTER

    # -----------------------------------------------------------------------
    # SLIDE 2: The Big Problem
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "🦠 The Big Problem: Why Do Viruses Escape Vaccines?", "Understanding viral mutation & immune evasion")

    # Card 1: ELI5 Analogy
    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), border_color=COLOR_AMBER)
    tb = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.0), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "👶 ELI5 Explanation (Simple Analogy)"
    p.font.name = FONT_NAME
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER

    p2 = tf.add_paragraph()
    p2.text = (
        "\nImagine a sneaky burglar (the Virus) wearing a yellow jacket.\n\n"
        "Security guards (Antibodies) learn to spot the yellow jacket and grab him.\n\n"
        "Next month, the burglar switches to a blue hoodie! Now guards walk right past him!\n\n"
        "👉 Our Goal: Predict what new coat the burglar will wear BEFORE he changes it!"
    )
    p2.font.name = FONT_NAME
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_WHITE

    # Card 2: Biological Reality
    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(7.1), Inches(2.0), Inches(5.1), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🧬 Biological Reality"
    p.font.name = FONT_NAME
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    p2 = tf.add_paragraph()
    p2.text = (
        "\n• SARS-CoV-2 Spike Protein has 1,273 amino acid positions.\n\n"
        "• Mutation Hotspots: Specific positions where mutations frequently occur.\n\n"
        "• Immune Escape: Mutations that prevent existing human antibodies from binding.\n\n"
        "• Challenge: Standard surveillance only detects mutations AFTER thousands of people are infected."
    )
    p2.font.name = FONT_NAME
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_WHITE

    # -----------------------------------------------------------------------
    # SLIDE 3: The Core Solution
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "💡 The Core Solution: Joint Mutation + Escape Prediction", "Coupling structural geometry with language intelligence")

    # Card 1: Mutation Hotspot AI
    add_card(slide, Inches(0.8), Inches(1.8), Inches(3.6), Inches(3.2), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(3.2), Inches(2.8))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "1. Hotspot AI (P)"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN
    tf.add_paragraph().text = "\nWill the virus change this spot without breaking its body structure?\n\nOutput: Probability P(mutate) ∈ [0, 1]"
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # Card 2: Immune Escape AI
    add_card(slide, Inches(4.8), Inches(1.8), Inches(3.6), Inches(3.2), border_color=COLOR_GREEN)
    tb = slide.shapes.add_textbox(Inches(5.0), Inches(2.0), Inches(3.2), Inches(2.8))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "2. Escape AI (S)"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_GREEN
    tf.add_paragraph().text = "\nWill changing this spot hide the virus from neutralizing antibodies?\n\nOutput: Escape Score S(escape) ∈ ℝ"
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # Card 3: Joint Risk Score Formula
    add_card(slide, Inches(8.8), Inches(1.8), Inches(3.7), Inches(3.2), border_color=COLOR_AMBER)
    tb = slide.shapes.add_textbox(Inches(9.0), Inches(2.0), Inches(3.3), Inches(2.8))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "3. Risk Score (R)"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_AMBER
    tf.add_paragraph().text = "\nJoint Risk Score Formula:\n\nR_i = P(mutate_i) × S(escape_i)\n\nRanks high-vulnerability sites for global health priority."
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # Bottom Callout Card
    add_card(slide, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5), border_color=COLOR_BLUE)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "🔑 Why Multimodal Fusion Matters"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN
    tf.add_paragraph().text = "Sequence alone misses 3D surface contact. Structure alone misses evolutionary language context. Cross-attention fuses both to capture true biological vulnerability!"
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # -----------------------------------------------------------------------
    # SLIDE 4: Public Data Sources Table
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "🌐 Public Data Sources & Databases", "100% open-access data — zero registration needed")

    # Table
    rows, cols = 6, 4
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(3.2)
    table.columns[2].width = Inches(3.8)
    table.columns[3].width = Inches(2.5)

    headers = ["Database", "URL / Source", "What it Provides", "Data Type / Shape"]
    data_rows = [
        ["🧬 Nextstrain Open", "data.nextstrain.org/files/ncov/open/", "Open SARS-CoV-2 viral sequences + date metadata", "FASTA + TSV (.zst)\n1,500 subsampled"],
        ["🧬 NCBI Virus", "ncbi.nlm.nih.gov/labs/virus", "GenBank SARS-CoV-2 Spike protein sequences", "Protein FASTA (.fasta)\n1273 aa ± 5"],
        ["🧪 Bloom Lab DMS", "github.com/jbloomlab/SARS2_RBD_Ab_escape_maps", "Deep Mutational Scanning experimental escape scores", "CSV (43,499 rows)\nTensor (1273, 5)"],
        ["🏛️ RCSB PDB", "rcsb.org (PDB: 6VXX)", "3D atomic coordinates of Spike protein trimer", "PDB file (.pdb)\nContact Graph (10 Å)"],
        ["🛡️ CoV-AbDab", "opig.stats.ox.ac.uk/webapps/covabdab", "Coronavirus Antibody Database (Class 1-4 lookup)", "CSV / Epitope Dictionary"]
    ]

    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_CARD_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_NAME
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_CYAN

    for r, row in enumerate(data_rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_DARK
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_NAME
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_TEXT_WHITE if c != 0 else COLOR_GREEN

    # -----------------------------------------------------------------------
    # SLIDE 5: Module 1 — Sequence AI (ESM-2)
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "🔤 Module 1: ESM-2 (Protein Language Model)", "Reading the 1D amino acid sentence of the virus")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "👶 ELI5: How ESM-2 Works"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN

    tf.add_paragraph().text = (
        "\n• Protein sequence is a 1,273-letter sentence: 'MFVFLVLLPL...'\n\n"
        "• ChatGPT reads English words; ESM-2 reads protein letters!\n\n"
        "• ESM-2 understands 'protein grammar' — if you change a letter, it knows if the protein sentence still makes sense."
    )
    tf.paragraphs[1].font.size = Pt(14)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), border_color=COLOR_GREEN)
    tb = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "📥 Technical Input / Output & Strategy"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_GREEN

    tf.add_paragraph().text = (
        "\n• Model: esm2_t33_650M_UR50D (650 Million parameters)\n\n"
        "• Input: Spike Sequence string (Length L = 1273)\n\n"
        "• Output: Feature Tensor H_seq ∈ ℝ^{L × 1280}\n\n"
        "• Strategy: Fully FROZEN. Embeddings extracted ONCE on Kaggle free GPU and saved as .pt tensor cache. Never recomputed during training!"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # -----------------------------------------------------------------------
    # SLIDE 6: Module 2 — 3D Structure AI (GVP-GNN)
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "🧩 Module 2: GVP-GNN (3D Graph Neural Network)", "Understanding the 3D origami shape of the virus")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "👶 ELI5: The 3D Spiderweb"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN

    tf.add_paragraph().text = (
        "\n• Proteins fold into 3D origami shapes in space.\n\n"
        "• Nodes (Dots): Each amino acid residue in 3D space.\n\n"
        "• Edges (Lines): Connected if 2 residues touch in 3D space (distance < 10 Å).\n\n"
        "• GVP (Geometric Vector Perceptron) tracks 3D angles and directions so rotational geometry is preserved."
    )
    tf.paragraphs[1].font.size = Pt(14)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), border_color=COLOR_GREEN)
    tb = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "📥 Graph Feature Specification"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_GREEN

    tf.add_paragraph().text = (
        "\n• Node Scalars (L × 10): SASA (surface area), dihedrals (sin/cos φ, ψ), secondary structure.\n\n"
        "• Node Vectors (L × 3 × 3): Backbone orientation vectors (N→CA, CA→C, CA→CB).\n\n"
        "• Edge Features (E × 2): Euclidean distance & sequence separation.\n\n"
        "• Output: Structural Tensor H_struct ∈ ℝ^{L × 512}"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # -----------------------------------------------------------------------
    # SLIDE 7: Module 3 — Cross-Attention Fusion
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "🔀 Module 3: Cross-Attention Fusion", "Connecting 1D sequence text with 3D structural context")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0), border_color=COLOR_BLUE)
    tb = slide.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(11.1), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "👶 ELI5: The Expert Conversation"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN

    tf.add_paragraph().text = (
        "\nImagine two experts discussing residue #484:\n"
        "• Expert 1 (ESM-2): 'Residue #484 has a high evolutionary mutation rate in text!'\n"
        "• Expert 2 (GVP-GNN): 'Let me check my 3D map... Yes! It sits right on the outer surface where Class 1 antibodies grab the virus!'\n\n"
        "👉 Cross-Attention allows sequence queries to attend directly to 3D structural keys & values."
    )
    tf.paragraphs[1].font.size = Pt(14)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    tf.add_paragraph().text = (
        "\nMathematical Formulation:\n"
        "  Q = Proj(H_seq) ∈ ℝ^{L × 512},   K = Proj(H_struct) ∈ ℝ^{L × 512},   V = Proj(H_struct) ∈ ℝ^{L × 512}\n"
        "  H_fused = LayerNorm(Q + MultiHeadAttention(Q, K, V))   → Output Shape: ℝ^{L × 512}"
    )
    tf.paragraphs[2].font.size = Pt(13)
    tf.paragraphs[2].font.color.rgb = COLOR_AMBER

    # -----------------------------------------------------------------------
    # SLIDE 8: Dual Heads & Joint Loss
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "🎯 Module 4: Dual Prediction Heads & Joint Loss", "Simultaneous mutation hotspot classification & escape regression")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "1. Hotspot Head (Sigmoid)"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN

    tf.add_paragraph().text = (
        "\n• Predicts mutation probability P(mutate_i) ∈ [0, 1]\n\n"
        "• Loss: Binary Cross-Entropy (BCE)\n\n"
        "• Metric: AUROC & AUPRC"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    tf.add_paragraph().text = "\n2. Escape Head (Linear)"
    tf.paragraphs[2].font.size = Pt(16)
    tf.paragraphs[2].font.bold = True
    tf.paragraphs[2].font.color.rgb = COLOR_GREEN

    tf.add_paragraph().text = (
        "\n• Predicts quantitative DMS escape score S(escape_i) ∈ ℝ\n\n"
        "• Loss: Mean Squared Error (MSE) vs Bloom Lab DMS\n\n"
        "• Metric: Spearman Rank Correlation (ρ)"
    )
    tf.paragraphs[3].font.size = Pt(13)
    tf.paragraphs[3].font.color.rgb = COLOR_TEXT_WHITE

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), border_color=COLOR_AMBER)
    tb = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "🧮 Joint Loss & Risk Formula"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_AMBER

    tf.add_paragraph().text = (
        "\nJoint Loss Function:\n"
        "  L = λ_1 · BCE(P_mutate, y_mutate) + λ_2 · MSE(S_escape, y_dms)\n\n"
        "Joint Risk Score Formula:\n"
        "  R_i = P(mutate_i) × S(escape_i)\n\n"
        "👉 Early Stopping Rule: Monitor Validation Spearman ρ (not loss!) to prevent rank degradation."
    )
    tf.paragraphs[1].font.size = Pt(14)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # -----------------------------------------------------------------------
    # SLIDE 9: Temporal Split Strategy
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "⏳ The Time-Machine Test: Strict Temporal Split", "Zero temporal leakage guarantee for true prospective validation")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "📅 Training & Validation Set"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN

    tf.add_paragraph().text = (
        "\n• Cutoff Date: December 31, 2023\n\n"
        "• Includes: Wuhan-Hu-1, Alpha, Delta, Omicron BA.1, BA.5 sequences + Bloom Lab DMS.\n\n"
        "• Automated Assertion in CI:\n"
        "  assert max(train.date) <= Cutoff <= min(test.date)"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), border_color=COLOR_GREEN)
    tb = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "🔮 Prospective Test Set (2024–2026)"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_GREEN

    tf.add_paragraph().text = (
        "\n• Emerged Variants: JN.1, KP.2, KP.3, XEC\n\n"
        "• Evaluation Metrics:\n"
        "  - Precision@5, Precision@10, Precision@20 on actual emerged mutations\n"
        "  - Spearman ρ rank correlation vs actual escape\n"
        "  - AUROC & AUPRC for hotspot classification"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # -----------------------------------------------------------------------
    # SLIDE 10: 100% Free Cloud Compute Strategy
    # -----------------------------------------------------------------------
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, COLOR_BG_DARK)
    add_header(slide, "⚡ 100% Free Cloud Compute Strategy", "Zero AWS/GCP cost — zero laptop hardware strain")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), border_color=COLOR_CYAN)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "💻 Local Laptop Role (0% Strain)"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_CYAN

    tf.add_paragraph().text = (
        "\n• Code editing in VS Code\n"
        "• Downloading raw files & dataset setup\n"
        "• Pure Pandas CSV filtering & temporal splits\n"
        "• Building 3D contact graphs locally (~5-15 mins)\n"
        "• Plotting result charts (matplotlib/seaborn)\n\n"
        "👉 Laptop NEVER runs heavy ESM-2 or model training!"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), border_color=COLOR_GREEN)
    tb = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    tf.paragraphs[0].text = "☁️ Kaggle Free Cloud (NVIDIA T4 16GB)"
    tf.paragraphs[0].font.size = Pt(18)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = COLOR_GREEN

    tf.add_paragraph().text = (
        "\n• MAFFT MSA alignment on Kaggle free CPU (~30 mins)\n"
        "• ESM-2 650M embedding extraction pass on Kaggle T4 (~1-2 hrs)\n"
        "• GVP-GNN + Cross-Attention model training (~1-2 hrs)\n"
        "• Full 4-condition ablation suite (~4-8 hrs total)\n\n"
        "👉 Total Project Usage: ~12-20 GPU-hrs out of 30 free GPU-hrs/week!"
    )
    tf.paragraphs[1].font.size = Pt(13)
    tf.paragraphs[1].font.color.rgb = COLOR_TEXT_WHITE

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"Presentation saved successfully -> {output_path}")


if __name__ == "__main__":
    out = Path("reports/Viral_Mutation_Escape_Prediction.pptx")
    build_presentation(out)
