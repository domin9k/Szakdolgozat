# DermAI Scan – Kép-előfeldolgozási folyamat (Pre-processing Pipeline)

Ez a dokumentum részletesen bemutatja, hogyan megy végbe a képek előfeldolgozása a **DermAI Scan** szoftverben a kép rögzítésétől kezdve egészen a mélytanulási modellek ([MobileNetV2](file:///Users/laszlodominik/Desktop/Projektmunka_2/mobilenet_skin_model.keras) és [EfficientNet](file:///Users/laszlodominik/Desktop/Projektmunka_2/efficientnet_skin_model.keras)) bemeneti rétegéig.

---

## 1. Kliensoldali előkészítés (`webapp/static/app.js`)

A folyamat a felhasználói felületen indul el, kétféle beviteli móddal:

* **Kamera használatakor (`capturePhoto`):**
  * A felhasználó előtt megjelenik egy célzónégyzet és egy ovális segédvonal a lezió optimális elhelyezéséhez.
  * A felvétel gomb megnyomásakor a rendszer a videófolyamból egy HTML5 `<canvas>` elemre rajzolja a képet.
  * A videó rövidebb oldalát (`min(videoWidth, videoHeight)`) alapul véve a kép közepéből egy négyzet alakú kivágást (`square crop`) hoz létre. Ezzel megakadályozza, hogy a képarány torzuljon a későbbi feldolgozás során.
  * A kivágott képet tömörített JPEG Blob formátumba alakítja (0.92 minőségi faktorral), és tárolja elküldéshez.

* **Fájlfeltöltéskor (`handleFileSelect`):**
  * Helyi fájl (JPEG, PNG, WebP) kiválasztása esetén a fájl közvetlenül betöltődik a memóriába, és megjelenik a vizuális előnézet a minőségellenőrzéssel párhuzamosan.

---

## 2. Kép fogadása és formátum-egységesítés (`webapp/server.py`)

A beérkező kérést a FastAPI kiszolgáló fogadja a `/api/predict` vagy `/api/quality-check` végponton:

1. **MIME-típus ellenőrzése:**
   * A szerver ellenőrzi a fejlécet: kizárólag `image/jpeg`, `image/png` és `image/webp` típusokat fogad el.
2. **RGB konverzió (Pillow):**
   * A bináris adatfolyamból megnyitott képet a Pillow könyvtár explicit módon átkonvertálja 3 csatornás RGB formátumra (`img.convert("RGB")`).
   * Ez kiküszöböli a PNG fájlok esetleges alfa (átlátszósági) csatornáját (RGBA) vagy a szürkeárnyalatos formátumok miatti tenzordimenzió-eltéréseket.

---

## 3. Automatikus képminőség-ellenőrzés (`assess_image_quality`)

Mielőtt a modell megkapná a képet, lefut egy algoritmusos minőség-ellenőrzés OpenCV segítségével. A rendszer 100 pontból indul ki, és hibák esetén pontlevonást alkalmaz:

| Szempont | Módszer / Metrika | Küszöbérték | Hiba / Levonás |
| :--- | :--- | :--- | :--- |
| **Felbontás** | Kép minimális kiterjedése (`min(w, h)`) | $\ge 100\text{ px}$ | Túl kicsi felbontás esetén $-40$ pont |
| **Életlenség (Blur)** | Szürkeárnyalatos kép Laplace-varianciája | $\text{Var}(\Delta I) \ge 80.0$ | Enyhe életlenségnél $-15$ pont, erős homályosságnál $-30$ pont |
| **Fényerő (Brightness)** | Szürkeárnyalatos kép átlaga | $40 \le \bar{I} \le 220$ | Túl sötét ($<40$) vagy kiégett/túl világos ($>220$) esetén $-20$ pont |
| **Kontraszt** | Szürkeárnyalatos értékek szórása ($\sigma$) | $\sigma \ge 20$ | Alacsony kontraszt esetén $-15$ pont |
| **Színtelítettség** | HSV színtér S-csatornájának átlaga | $\bar{S} \ge 10$ | Szürkeárnyalatos vagy nem bőrt ábrázoló képnél $-10$ pont |

* **Döntés:** Ha a minőségi pontszám eléri vagy meghaladja a 40 pontot, a kép elfogadható (`is_acceptable = True`). 40 pont alatt a szerver **HTTP 422** állapotkóddal elutasítja a vizsgálatot, és visszaküldi a felhasználónak a konkrét javaslatokat (pl. „Fényképezze újra élesebb fókusszal”, „Használjon jobb megvilágítást”).

---

## 4. Geometriai és tenzor transzformációk (`preprocess_for_model`)

Amennyiben a minőség megfelelő, a kép a konkrét modell-előkészítő csővezetékbe kerül:

1. **Központi négyzetes vágás (`square_crop_center`):**
   * Ha a feltöltött kép nem négyzetes (pl. közvetlen fájlfeltöltésből származik), a szerver kivágja a legnagyobb központi négyzetet:
     $$\text{side} = \min(w, h), \quad \text{left} = \frac{w - \text{side}}{2}, \quad \text{top} = \frac{h - \text{side}}{2}$$
2. **Átméretezés (Resize 224×224):**
   * A képet a neurális hálók bemeneti méretére méretezi át magas minőségű Lanczos interpolációval (`Image.LANCZOS`):
     $$(w, h) \to (224, 224)$$
3. **Tömb- és Batch-dimenzió képzés:**
   * Lebegőpontos NumPy tömbbé alakítás: `np.array(img, dtype=np.float32)`
   * Batch dimenzió hozzáadása: a kapott tenzor alakja `(1, 224, 224, 3)`.

---

## 5. Modell-specifikus normalizáció

A különböző architektúrák tanítása eltérő skálázású bemeneteken történt, ezért a csővezeték modell szerint ágazik szét:

* **MobileNetV2 (`preprocess_mobilenet`):**
  * A képpontok értékeit a $[-1, 1]$ intervallumra transzformálja:
    $$x_{\text{norm}} = \frac{x}{127.5} - 1.0$$
* **EfficientNet (`preprocess_efficientnet`):**
  * A képpontokat a $[0, 1]$ vagy standard skálára normalizálja (belsőleg osztás 255-tel).

---

## 6. Test-Time Augmentation (TTA – `predict_with_tta`)

Az **EfficientNet** esetében a mélyebb, robusztusabb diagnózis érdekében kiegészítő predikció-közbeni adatbővítés zajlik (alapértelmezetten 5 lépés):

1. **1. lépés:** Eredeti, módosítatlan kép predikciója.
2. **További lépések:** Szabályos tükrözések alkalmazása a tenzoron:
   * Vízszintes tükrözés: `np.flip(augmented, axis=2)`
   * Függőleges tükrözés: `np.flip(augmented, axis=1)`
3. **Összegzés:** A futtatott predikciók valószínűségi értékeinek számtani átlagát képzi:
   $$P_{\text{final}} = \frac{1}{N} \sum_{i=1}^{N} P_i$$

---

## Összefoglaló folyamatábra

```
[Kamera / Fájlfeltöltés]
           │
           ▼
[Központi négyzetes vágás (Kliens / Szerver)]
           │
           ▼
[RGB konverzió (3 csatorna)]
           │
           ▼
[Minőségellenőrzés: Élesség, Fényerő, Kontraszt, Telítettség]
           │
      Elfogadva? ─── Nem ───► [HTTP 422: Figyelmeztetések visszaküldése]
           │ Igen
           ▼
[Átméretezés 224x224-re (Lanczos)]
           │
           ▼
[Tenzorrá alakítás: shape = (1, 224, 224, 3)]
           │
   ┌───────┴────────────────────────┐
   ▼                                ▼
[MobileNetV2 skálázás: [-1, 1]]   [EfficientNet skálázás: [0, 1]]
   │                                │
   ▼                                ▼
[Gyors szűrés predikció]          [TTA adatbővítés (Tükrözések)]
                                    │
                                    ▼
                                  [Mély analízis átlagolt predikció]
```
