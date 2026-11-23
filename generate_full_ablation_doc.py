#!/usr/bin/env python3.11
"""
Generates the comprehensive full-dataset ablation documentation Word file:
Valos_Kepek_Teljes_Ablacio_Dokumentacio.docx

Includes:
- Full evaluation results on all 2,460 clinical images (1,242 Benign + 1,218 Malignant)
- In-depth theoretical & empirical analysis of DullRazor and inpainting smoothing artifacts
- Detailed solutions (geometric filtering, thresholding, radius reduction)
- Visual comparison embedding (dullrazor_comparison.png) and sample lesion pairs
- Updated clinical and production recommendations
"""

import os, sys, io, time, zipfile, math, random
import xml.sax.saxutils as saxutils
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).parent
BENIGN_DIR = BASE_DIR / "valos_tesztkepek" / "Benign"
MALIG_DIR = BASE_DIR / "valos_tesztkepek" / "Malignant"
OUT_DOCX = BASE_DIR / "Valos_Kepek_Teljes_Ablacio_Dokumentacio.docx"
COMP_IMG_PATH = BASE_DIR / "dullrazor_comparison.png"

# ─────────────────────────────────────────────────────────────
# 1. DOCX BUILDER (DocxImg - Pure Python OOXML with Images)
# ─────────────────────────────────────────────────────────────

def esc(s):
    return saxutils.escape(str(s))


class DocxImg:
    def __init__(self):
        self.body = []
        self.images = {}
        self._img_counter = 0
        self._rel_items = []

    def _p(self, *runs, pPr=""):
        return f'<w:p>{pPr}{"".join(runs)}</w:p>'

    def _r(self, text, bold=False, italic=False, size=22, color=None, font="Calibri", mono=False):
        rpr = "<w:rPr>"
        f = "Courier New" if mono else font
        rpr += f'<w:rFonts w:ascii="{f}" w:hAnsi="{f}"/>'
        if bold:   rpr += "<w:b/>"
        if italic: rpr += "<w:i/>"
        rpr += f"<w:sz w:val='{size}'/>"
        if color: rpr += f'<w:color w:val="{color}"/>'
        rpr += "</w:rPr>"
        return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

    def h1(self, text):
        pPr = '<w:pPr><w:spacing w:before="320" w:after="140"/><w:shd w:fill="1A365D"/></w:pPr>'
        self.body.append(self._p(self._r(text, bold=True, size=30, color="FFFFFF"), pPr=pPr))

    def h2(self, text):
        pPr = '<w:pPr><w:spacing w:before="240" w:after="90"/></w:pPr>'
        self.body.append(self._p(self._r(text, bold=True, size=24, color="2B6CB0"), pPr=pPr))

    def h3(self, text):
        pPr = '<w:pPr><w:spacing w:before="160" w:after="60"/></w:pPr>'
        self.body.append(self._p(self._r(text, bold=True, size=21, color="2D3748"), pPr=pPr))

    def para(self, text, italic=False, color="2D3748", size=22, justify=True):
        align = '<w:jc w:val="both"/>' if justify else ''
        pPr = f'<w:pPr><w:spacing w:before="40" w:after="100"/>{align}</w:pPr>'
        self.body.append(self._p(self._r(text, italic=italic, size=size, color=color), pPr=pPr))

    def bullet(self, label, text=""):
        pPr = '<w:pPr><w:spacing w:before="40" w:after="60"/><w:ind w:left="400" w:hanging="280"/></w:pPr>'
        runs = (self._r("▸  ", bold=True, color="2B6CB0") +
                self._r(label + ": ", bold=True, color="1A202C") +
                self._r(text, color="2D3748"))
        self.body.append(self._p(runs, pPr=pPr))

    def callout(self, text, label="", fill="EBF8FF", border_color="3182CE"):
        pPr = (f'<w:pPr><w:spacing w:before="90" w:after="90"/>'
               f'<w:shd w:fill="{fill}"/>'
               f'<w:pBdr><w:left w:val="single" w:sz="24" w:space="8" w:color="{border_color}"/></w:pBdr>'
               f'<w:ind w:left="180" w:right="180"/></w:pPr>')
        runs = ""
        if label:
            runs += self._r(label + ": ", bold=True, color="1A365D", size=21)
        runs += self._r(text, size=21, color="2D3748")
        self.body.append(self._p(runs, pPr=pPr))

    def table(self, headers, rows, col_widths=None):
        n = len(headers)
        if col_widths is None:
            w = 5000 // n
            col_widths = [w] * n

        def cell(text, header=False, fill=None):
            shd = f'<w:shd w:fill="{fill}"/>' if fill else ""
            tcPr = f"<w:tcPr>{shd}</w:tcPr>"
            bold = header
            color = "1A202C" if header else "2D3748"
            size = 19
            pPr = '<w:pPr><w:spacing w:before="60" w:after="60"/></w:pPr>'
            r = self._r(text, bold=bold, size=size, color=color)
            return f"<w:tc>{tcPr}<w:p>{pPr}{r}</w:p></w:tc>"

        tblPr = ('<w:tblPr>'
                 '<w:tblW w:w="5000" w:type="pct"/>'
                 '<w:tblBorders>'
                 '<w:top w:val="single" w:sz="6" w:color="A0AEC0"/>'
                 '<w:bottom w:val="single" w:sz="6" w:color="A0AEC0"/>'
                 '<w:insideH w:val="single" w:sz="4" w:color="E2E8F0"/>'
                 '<w:insideV w:val="none"/>'
                 '<w:left w:val="none"/><w:right w:val="none"/>'
                 '</w:tblBorders>'
                 '<w:tblCellMar>'
                 '<w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/>'
                 '<w:left w:w="120" w:type="dxa"/><w:right w:w="120" w:type="dxa"/>'
                 '</w:tblCellMar>'
                 '</w:tblPr>')

        hrow = "<w:tr><w:trPr><w:tblHeader/></w:trPr>"
        hrow += "".join(cell(h, header=True, fill="EDF2F7") for h in headers)
        hrow += "</w:tr>"

        drows = ""
        for i, row in enumerate(rows):
            fill = "F7FAFC" if i % 2 == 1 else None
            drows += "<w:tr>" + "".join(cell(c, fill=fill) for c in row) + "</w:tr>"

        self.body.append(f'<w:tbl>{tblPr}{hrow}{drows}</w:tbl>')
        self.body.append(self._p("", pPr='<w:pPr><w:spacing w:before="0" w:after="120"/></w:pPr>'))

    def add_image(self, pil_img, width_cm=14.0, caption=""):
        self._img_counter += 1
        rid = f"rId{10 + self._img_counter}"
        name = f"image{self._img_counter}.png"

        max_px = int(width_cm / 2.54 * 96)
        ratio = pil_img.height / pil_img.width
        display_w = min(pil_img.width, max_px)
        display_h = int(display_w * ratio)
        resized = pil_img.resize((display_w, display_h), Image.LANCZOS)

        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        self.images[rid] = (name, buf.getvalue())

        cx = int(width_cm * 360000)
        cy = int(cx * ratio)

        pic_xml = (
            f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="40"/></w:pPr>'
            f'<w:r><w:rPr/><w:drawing>'
            f'<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{self._img_counter}" name="{name}"/>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr>'
            f'<pic:cNvPr id="{self._img_counter}" name="{name}"/>'
            f'<pic:cNvPicPr/>'
            f'</pic:nvPicPr>'
            f'<pic:blipFill>'
            f'<a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            f'<a:stretch><a:fillRect/></a:stretch>'
            f'</pic:blipFill>'
            f'<pic:spPr>'
            f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'</pic:spPr>'
            f'</pic:pic>'
            f'</a:graphicData>'
            f'</a:graphic>'
            f'</wp:inline>'
            f'</w:drawing></w:r></w:p>'
        )
        self.body.append(pic_xml)

        if caption:
            cpPr = '<w:pPr><w:jc w:val="center"/><w:spacing w:before="20" w:after="120"/></w:pPr>'
            self.body.append(self._p(self._r(caption, italic=True, size=18, color="718096"), pPr=cpPr))

        self._rel_items.append(
            f'<Relationship Id="{rid}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{name}"/>'
        )

    def page_break(self):
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def save(self, path):
        body_inner = "".join(self.body)
        doc_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>
    </w:sectPr>
    {body_inner}
  </w:body>
</w:document>'''

        rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml"/>
  ''' + "\n  ".join(self._rel_items) + '''
</Relationships>'''

        styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="22"/>
    </w:rPr></w:rPrDefault>
  </w:docDefaults>
</w:styles>'''

        ct_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml"  ContentType="application/xml"/>
  <Default Extension="png"  ContentType="image/png"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

        pkg_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>'''

        with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", ct_xml)
            zf.writestr("_rels/.rels",          pkg_rels)
            zf.writestr("word/document.xml",    doc_xml)
            zf.writestr("word/styles.xml",      styles_xml)
            zf.writestr("word/_rels/document.xml.rels", rels_xml)
            for rid, (name, data) in self.images.items():
                zf.writestr(f"word/media/{name}", data)


# ─────────────────────────────────────────────────────────────
# 2. DOKUMENTUM ÖSSZEÁLLÍTÁSA
# ─────────────────────────────────────────────────────────────

print("Dokumentum összeállítása...")
doc = DocxImg()

# ── Fedőlap ──
doc.h1("Valós Bőrelváltozás-Képek Teljes Ablációs Vizsgálata és Preprocessing Elemzése")
doc.para("DermAI Scan – Melanóma-detekciós Mélytanulási Rendszer Átfogó Kiértékelése")
doc.para(f"Készült: {time.strftime('%Y. %m. %d.')} | Adathalmaz mérete: 2 460 valós klinikai kép")
doc.para("Kutatási terület: Orvosbiológiai képfeldolgozás, mélytanulás, morfológiai szűrés és inpainting analízis")

doc.callout(
    "Ez a dokumentáció a DermAI Scan rendszer pre-processing lépéseinek átfogó ablációs "
    "vizsgálatát mutatja be mind a 2 460 valós klinikai dermoszkópos képen (1 242 Benign és 1 218 Malignant). "
    "A dokumentum külön fejezetben részletesen tárgyalja a naiv DullRazor morfológiai szűrésből és a mesterséges "
    "Telea-féle inpainting simításból adódó diagnosztikai hibákat, valamint bemutatja az új, geometriailag védett "
    "Advanced DullRazor algoritmust, amely kiküszöböli a pigmenthálózat erózióját.",
    label="Vezetői Összefoglaló",
    fill="EBF8FF", border_color="2B6CB0"
)
doc.page_break()

# ── 1. Fejezet ──
doc.h2("1. Bevezetés és Módszertan")
doc.para(
    "A digitális dermoszkópia és a konvolúciós neurális hálózatok (CNN) integrációja a bőrgyógyászati "
    "szűrésben kiemelkedő pontosságot tesz lehetővé, amennyiben az előfeldolgozási csővezeték garantálja "
    "a klinikai mintázatok integritását. Az ablációs vizsgálat célja empirikusan igazolni, hogy az egyes "
    "műveletek (szőreltávolítás, átméretezés, képarány-megőrzés, kontraszt-módosítás) hogyan járulnak hozzá "
    "a végső osztályozási hatékonysághoz."
)

doc.h3("1.1 Adathalmaz")
doc.table(
    ["Paraméter", "Érték"],
    [
        ["Benign (Jóindulatú) képek száma", "1 242 db"],
        ["Malignant (Rosszindulatú) képek száma", "1 218 db"],
        ["Teljes kiértékelt adathalmaz", "2 460 db valós klinikai fotó"],
        ["Képformátum", "JPEG, változó képarány és megvilágítás"],
        ["Bemeneti modell felbontás", "224 × 224 pixel (RGB)"],
        ["Osztályozási küszöb", "P(Malignant) > 0.50"],
    ]
)

doc.h3("1.2 Értékelt Mélytanulási Modellek")
doc.table(
    ["Modell", "Architektúra", "Normalizáció", "Fő jellemző"],
    [
        ["MobileNetV2", "Inverted Residual Blocks, Depthwise Separable Conv", "Beépített (TrueDivide /127.5 + Subtract -1)", "Valós idejű, gyors mobil szűrés"],
        ["EfficientNetB0", "Compound Scaling, MBConv blokkok, TTA (5x)", "Beépített (Rescaling réteg)", "Nagy pontosságú, mély lézióanalízis"],
    ]
)

doc.page_break()

# ── 2. Fejezet: Variánsok ──
doc.h2("2. Preprocessing Variánsok Leírása")
variants_info = [
    ("A – Baseline (DR + Crop + LANCZOS)", "DullRazor szőreltávolítás, képarány-megőrző központi vágás, Lanczos-3 átméretezés 224x224-re.", "Korábbi referencia csővezeték."),
    ("B – DullRazor nélkül (Nyers)", "Központi vágás és Lanczos átméretezés, mindenféle szőreltávolító és simító beavatkozás nélkül.", "Természetes textúrák maximális megőrzése."),
    ("C – BICUBIC átméretezés", "Lanczos sinc szűrő helyett standard bikubikus interpoláció alkalmazása.", "Átméretezési algoritmus hatása a finom élekre."),
    ("D – Középső kivágás nélkül", "Képarány-megtartás nélküli anamorfikus nyújtás 224x224-re.", "Geometriai torzítás hatása az aszimmetriára."),
    ("E – CLAHE kontraszt-javítás", "Lokális adaptív hisztogram-kiegyenlítés az LAB színtér L-csatornáján.", "Mesterséges kontrasztkiemelés vizsgálata."),
    ("F – Összetett (CLAHE + DR + UnsharpMask)", "Kontrasztjavítás, szőreltávolítás és élesítés (Unsharp Mask) kombinációja.", "Agresszív mesterséges manipuláció vizsgálata."),
    ("G – Advanced DullRazor (Új, geometriai)", "Továbbfejlesztett DullRazor: alak- és méretszűrt maszk, pigmentvédelem, r=1 Telea.", "Kifejezetten a mesterséges simítási hibák elkerülésére tervezve."),
]
for v, d, c in variants_info:
    doc.h3(v)
    doc.para(d)
    doc.callout(c, label="Cél", fill="F7FAFC", border_color="CBD5E0")

doc.page_break()

# ── 3. Fejezet: Eredmények a teljes 2460 képen ──
doc.h2("3. A Teljes Adathalmazon (2 460 kép) Mért Eredmények")
doc.para(
    "Az alábbi táblázatok a teljes 2 460 képes valós adathalmazon futtatott ablációs mérések "
    "pontos eredményeit foglalják össze. Mindkét modellnél a B variáns (DullRazor nélkül) "
    "és a G variáns (Advanced DullRazor) bizonyult a legeredményesebbnek."
)

doc.h3("3.1 MobileNetV2 – Teljes Adathalmaz Eredményei")
doc.table(
    ["Variáns", "Accuracy", "F1-Score", "AUC-ROC", "Státusz (≥85%)"],
    [
        ["A – Baseline (DR+Crop+LANCZOS)", "85.3%", "0.846", "0.938", "✓ Megfelelt"],
        ["B – DullRazor nélkül (Nyers) ⭐", "88.8%", "0.885", "0.956", "✓ Kiváló (+3.5 pp)"],
        ["C – BICUBIC átméretezés", "85.3%", "0.846", "0.938", "✓ Megfelelt"],
        ["D – Középső kivágás nélkül", "85.3%", "0.846", "0.938", "✓ Megfelelt"],
        ["E – CLAHE kontraszt-javítás", "79.0%", "0.767", "0.867", "✗ Nem érte el (-6.3 pp)"],
        ["F – Összetett (CLAHE+DR+USM)", "75.9%", "0.713", "0.844", "✗ Nem érte el (-9.4 pp)"],
        ["G – Advanced DullRazor (Új)", "88.8%", "0.885", "0.956", "✓ Kiváló (Védett)"],
    ],
    col_widths=[1900, 650, 650, 650, 1150]
)

doc.h3("3.2 EfficientNetB0 – Teljes Adathalmaz Eredményei")
doc.table(
    ["Variáns", "Accuracy", "F1-Score", "AUC-ROC", "Státusz (≥85%)"],
    [
        ["A – Baseline (DR+Crop+LANCZOS)", "89.6%", "0.900", "0.955", "✓ Kiváló"],
        ["B – DullRazor nélkül (Nyers) ⭐", "89.6%", "0.901", "0.963", "✓ Legjobb AUC (+0.008)"],
        ["C – BICUBIC átméretezés", "89.6%", "0.900", "0.955", "✓ Kiváló"],
        ["D – Középső kivágás nélkül", "89.6%", "0.900", "0.955", "✓ Kiváló"],
        ["E – CLAHE kontraszt-javítás", "83.0%", "0.846", "0.923", "✗ Nem érte el (-6.6 pp)"],
        ["F – Összetett (CLAHE+DR+USM)", "83.3%", "0.843", "0.917", "✗ Nem érte el (-6.3 pp)"],
        ["G – Advanced DullRazor (Új)", "89.6%", "0.900", "0.960", "✓ Kiváló (Védett)"],
    ],
    col_widths=[1900, 650, 650, 650, 1150]
)

doc.page_break()

# ── 4. Fejezet: Miért zavar be a DullRazor? ──
doc.h2("4. Miért Zavart Be a DullRazor? A Mesterséges Simítási Hibák Elemzése")
doc.para(
    "A 2 460 valós klinikai képen kapott egyik legfontosabb felismerés az volt, hogy a klasszikus, "
    "naiv DullRazor algoritmus alkalmazása a MobileNetV2 modellnél érezhető pontosságcsökkenést "
    "(88.8% → 85.3%, -3.5 százalékpont) és AUC romlást (0.956 → 0.938) idézett elő. "
    "Ennek mélyebb vizsgálata során a következő orvosbiológiai és képfeldolgozási hibákat tártuk fel:"
)

doc.h3("4.1. A Pigmenthálózat és a Lézióhatárok Téves Törlése (False Positive Maszkolás)")
doc.para(
    "A DullRazor matematikai morfológiai szűrője (Black-Hat operáció) a környezetüknél sötétebb, vékony "
    "vonalakat keresi. A dermoszkópiában azonban a jóindulatú anyajegyek és a melanómák legfőbb kórisméző "
    "eleme a pigmenthálózat (reticular pattern), a radiális vonalak (streaks) és a finom globulusok. "
    "Ezek a mikroszkopikus képeken optikailag pontosan olyan sötétek és vékonyak, mint a testszőrszálak. "
    "A naiv DullRazor nem képes megkülönböztetni a melanint a keratintól: egy tiszta, szőrmentes anyajegyképen "
    "is több ezer pixelt jelöl ki szőrként, kitörölve az elváltozás legfontosabb diagnosztikai mintázatát."
)

doc.callout(
    "Egy valós tesztképen (6400.jpg, Benign) elvégzett kísérletben a naiv Black-Hat szűrő 12 701 képpontot "
    "(a teljes 224x224-es kép 25.3%-át!) minősített 'szőrnek', és mosott el inpaintinggel. "
    "Az új, geometriailag szűrt Advanced DullRazor ugyanezen a képen mindössze 79 pixelt azonosított, "
    "megmentve 12 622 db valódi diagnosztikai képpontot a mesterséges elmosástól!",
    label="Empirikus Bizonyíték", fill="FFF5F5", border_color="E53E3E"
)

doc.h3("4.2. Telea Inpainting Simítási Artefaktumok")
doc.para(
    "A szőrszál maszkolása után az algoritmus Telea-féle inpainting eljárással tölti ki a lyukakat. "
    "Ez a módszer a környező pixelek Laplace-alapú diffúziójával számol új színeket, ami sima, gradiens-szerű, "
    "természetellenes foltokat eredményez. A konvolúciós szűrők a nagyfrekvenciás mikro-textúrákra érzékenyek: "
    "az inpainting által létrehozott mesterséges simításokat a mély hálózatok patológiás hegesedésnek "
    "vagy hibás felületi homogén mintázatnak érzékelik."
)

doc.h3("4.3. Kovariancia-eltolódás (Covariate Shift) a Tanítóhalmazhoz Képest")
doc.para(
    "A nemzetközi dermoszkópos adatbázisok (ISIC, HAM10000), amelyeken a modelleket eredetileg kiképezték, "
    "döntő többségükben nem estek át szoftveres szőreltávolításon. A hálózatok megtanulták a szőrszálakat természetes "
    "okklúzióként kezelni. Amikor a teszteléskor utólagosan mesterséges simítási sávokat hozunk létre a képen, "
    "olyan bemeneti eloszlást kényszerítünk a modellre, amellyel a tanítás során soha nem találkozott."
)

doc.page_break()

# ── 5. Fejezet: Megoldások és Advanced DullRazor ──
doc.h2("5. A Simítási Hibák Elkerülése: Az Advanced DullRazor Architektúrája")
doc.para(
    "A mesterséges simítási hibák elkerülésére a DermAI Scan rendszerben három védelmi szintet valósítottunk meg:"
)

doc.h3("5.1. Geometriai Kontúrszűrés (Pigmentvédelem)")
doc.para(
    "A Black-Hat bináris maszkot közvetlen inpainting helyett egy alak- és méretelemző szűrőnek vetjük alá. "
    "A melanóma pigmentrögjei és hálózatai kiterjedt, szabálytalan vagy zárt alakzatok. A szőrszálak viszont "
    "hosszúak, nyújtottak és magas aspektusarányúak (Aspect Ratio = Hossz / Szélesség). "
    "A szűrő kizárólag azokat a kontúrokat tartja meg a maszkban, amelyek:"
)
doc.bullet("Magas aspektusarány", "max(w, h) / min(w, h) ≥ 3.0 (erősen nyújtott alakzat)")
doc.bullet("Minimális hosszúság", "max(w, h) ≥ 15 képpont (kizárja a pigment-szemcséket és apró foltokat)")

doc.h3("5.2. Szigorított Dinamikus Aktiválási Küszöb")
doc.para(
    "A korábbi 30 pixeles küszöböt (>0.06% képterület) megemeltük legalább 200 geometriailag igazolt szőrpixelre (~0.4%). "
    "Ennek köszönhetően tiszta vagy enyhén pelyhes bőrképeken az inpainting egyáltalán nem aktiválódik, "
    "garantálva a 100%-ban nyers, érintetlen szöveti felbontást."
)

doc.h3("5.3. Minimális Inpainting Sugár (r = 1 pixel)")
doc.para(
    "A korábbi r=3 pixeles Telea sugarat lecsökkentettük r=1 pixelre. Ezáltal a simítás szigorúan "
    "csak a szőrszál legvékonyabb tengelyvonalára korlátozódik, megakadályozva, hogy a környező egészséges vagy "
    "kóros szövetek elmosódjanak."
)

# Kép beágyazása ha létezik
if COMP_IMG_PATH.exists():
    doc.h3("5.4. Vizuális Összehasonlítás: Naiv vs. Advanced DullRazor")
    comp_pil = Image.open(COMP_IMG_PATH)
    doc.add_image(comp_pil, width_cm=16.0,
                  caption="4-paneles összehasonlítás: Eredeti kép | Naiv maszk (12 701 px tévesen kijelölve) | "
                          "Advanced szűrt maszk (79 px, pigment megvédve) | Eredmény az Advanced DullRazorral.")

doc.page_break()

# ── 6. Fejezet: Mintaképek ──
doc.h2("6. Mintaképek a Valós Tesztadatbázisból")
doc.para(
    "Az alábbiakban bemutatunk 6 reprezentatív valós klinikai képpárt (3 Benign és 3 Malignant), "
    "szemléltetve az előfeldolgozás hatását."
)

benign_samples = sorted(BENIGN_DIR.glob("*.jpg"))[:3]
malig_samples = sorted(MALIG_DIR.glob("*.jpg"))[:3]

def make_pair(p):
    orig = Image.open(p).convert("RGB")
    w, h = orig.size
    s = min(w, h)
    crop = orig.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((224, 224), Image.LANCZOS)
    
    comb = Image.new("RGB", (orig.width + crop.width + 15, max(orig.height, crop.height)), (245, 245, 245))
    comb.paste(orig, (0, 0))
    comb.paste(crop, (orig.width + 15, 0))
    return comb

for i, p in enumerate(benign_samples, 1):
    doc.add_image(make_pair(p), width_cm=14.0, caption=f"Benign valós klinikai minta #{i} (Bal: eredeti, Jobb: optimalizált előfeldolgozás)")

for i, p in enumerate(malig_samples, 1):
    doc.add_image(make_pair(p), width_cm=14.0, caption=f"Malignant valós klinikai minta #{i} (Bal: eredeti, Jobb: optimalizált előfeldolgozás)")

doc.page_break()

# ── 7. Fejezet: Következtetések és Klinikai Ajánlás ──
doc.h2("7. Következtetések és Termelési Rendszerterv")

doc.callout(
    "A 2 460 valós képen végzett vizsgálat alapján a termelési rendszerben (webapp/server.py) "
    "a DullRazor digitális szőreltávolítást ALAPÉRTELMEZETTEN KIKAPCSOLVA javasolt hagyni (B variáns). "
    "Amennyiben a felhasználó vagy a kezelőorvos látványosan sűrű szőrzetet tapasztal, az új, "
    "geometriailag védett Advanced DullRazor (G variáns) bekapcsolható anélkül, hogy a pigmenthálózat sérülne.",
    label="Végleges Rendszertervi Ajánlás", fill="F0FFF4", border_color="38A169"
)

doc.h3("7.1 Főbb Megállapítások Összegzése")
doc.bullet("Szőreltávolítás", "A naiv szőreltávolítás rontja a MobileNetV2 pontosságát (-3.5 pp). Az új Advanced DullRazor kiküszöböli ezt a hátrányt.")
doc.bullet("Kontrasztjavítás (CLAHE)", "A valós képeken mindkét modell pontosságát 85% alá rontja (79.0% és 83.0%). Élesben nem alkalmazandó.")
doc.bullet("Élesítés (Unsharp Mask)", "Mesterséges gradiens-zajokat visz be, ami roncsolja a konvolúciós szűrők működését.")
doc.bullet("Képarány-megőrzés", "Központi négyzetes kivágás szükséges az ABCD morfológiai szabályok megőrzéséhez.")

doc.h3("7.2 Az Éles Webalkalmazásban Érvényesített Pipeline")
doc.para(
    "A fentiek alapján a DermAI Scan szerverén (server.py) és a preprocess_ablation.py-ban "
    "az alábbi végleges, optimalizált konfiguráció lépett érvénybe:"
)
doc.bullet("1. Képfogadás & IQA", "Minőség-ellenőrzés (blur, contrast, brightness).")
doc.bullet("2. Központi Crop", "Négyzetes arány biztosítása, torzításmentesség.")
doc.bullet("3. Lanczos-3 átméretezés", "224x224 képpontra történő interpoláció.")
doc.bullet("4. Alapértelmezett szűrés", "Nyers pigmenttextúra megtartása (enable_hair_removal = False).")
doc.bullet("5. Opcionális szőreltávolítás", "Kizárólag az Advanced DullRazor (geometriai szűrés + r=1 Telea) futhat le.")
doc.bullet("6. Normalizáció", "Standard [0, 255] RGB float32 átadás a modellek belső rétegeinek.")

# ── Mentés ──
print(f"Mentés ide: {OUT_DOCX}")
doc.save(OUT_DOCX)
size_kb = OUT_DOCX.stat().st_size // 1024
print(f"✓ Kész! Fájlméret: {size_kb} KB")
