---
name: "opendataloader-bbox-to-pymupdf-clip"
description: "opendataloader-pdf JSONが出力したバウンディングボックスを使ってPDFから画像を切り抜く・再レンダリングする場合に使用する。PDF座標系のバウンディングボックスをPyMuPDFのclip矩形へ変換し、Y軸反転の処理や高DPI画像クロップのレンダリングをカバーする。"
---

# OpenDataLoader BBox To PyMuPDF Clip

以下のような場合にこのスキルを使用する:

- `opendataloader-pdf` JSONをもとにPDFから画像を再切り抜きしたい
- 同じ領域を高DPIで再レンダリングして画質を向上させたい
- `opendataloader-pdf` の `bounding box` 値をPyMuPDFの `clip` に変換したい

## 基本ルール

`opendataloader-pdf` JSONの `bounding box` はPDF座標系で格納されている。
PyMuPDFの `get_pixmap(..., clip=...)` は左上原点の矩形を期待する。

ページ高さを使ってY軸を反転させる必要がある。

## 変換式

`bbox = [x0, y0, x1, y1]`、`page_height = page.rect.height` の場合:

```python
import fitz

def pdf_bbox_to_fitz_rect(bbox, page_height):
    x0, y0, x1, y1 = bbox
    return fitz.Rect(x0, page_height - y1, x1, page_height - y0)
```

これが核となる変換処理。座標系が既に変換済みであることを確認できない限り、生の `bounding box` を直接 `fitz.Rect(...)` に渡してはいけない。

## 高DPIクロップのパターン

```python
import fitz

doc = fitz.open("input.pdf")
page = doc[page_number - 1]
clip = pdf_bbox_to_fitz_rect(bbox, page.rect.height)
matrix = fitz.Matrix(scale, scale)
pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
pixmap.save("crop.png")
```

推奨する初期値:

- `scale=2`: 簡易確認向け
- `scale=3`: より高品質
- `scale=4` 以上: ファイルサイズと処理時間が許容できる場合のみ

## JSONで参照するフィールド

`opendataloader-pdf` の典型的な画像ノードは以下の構造を持つ:

```json
{
  "type": "image",
  "page number": 5,
  "bounding box": [45.374, 224.281, 537.265, 498.499],
  "source": "extracted_images/imageFile1.png"
}
```

使用するフィールド:

- `type == "image"`: 画像領域を特定する
- `page number`: PyMuPDFのページを選択する
- `bounding box`: `clip` を構築する
- `source`: 高解像度クロップ再生成時にファイル名を保持する

## ワークフロー

1. `opendataloader-pdf` JSONを読み込む。
2. `type == "image"` のノードを抽出する。
3. PyMuPDFで元のPDFを開く。
4. 各画像ノードに対して:
   `page number` を使ってページを取得する
   Y軸反転式で `bounding box` を変換する
   `fitz.Matrix(scale, scale)` でレンダリングする
   新しい出力ディレクトリに保存する
5. ユーザーが明示的に置き換えを求めない限り、元の抽出画像はそのまま保持する。

## 検証

コードの実装・編集後:

1. 既知の画像ノード1つを実際のクロップ結果と比較する。
2. クロップが垂直方向にずれている場合は、Y軸反転が省略されていないか、または二重に適用されていないか確認する。
3. 出力の寸法が `scale` にほぼ比例して拡大されていることを確認する。

## リポジトリ備考

このリポジトリでは、同じ変換パターンが以下に存在する:

- `paper_translator/renderers.py` -> `pdf_bbox_to_fitz_rect`

新たな座標変換を独自に実装するより、このロジックを再利用・参照することを優先する。
