# Design tokens (source of truth: figkit/style.py)

Derived from the group presentation: `build_ppt.py` L36–44 and the v4 deck shape XML. The decks were inspected on 2026-10-09.

## Colour roles: a fixed meaning, learned once

| Role | Hex | Meaning in every figure | Tint (card fill) |
|---|---|---|---|
| NAVY | `#1F4E79` | Structure, the DOS model, decisions that act on the screen | `#DDEAF6` |
| BLUE | `#2E6CA4` | The screen (descriptors, gates) | `#DDEAF6` |
| TEAL | `#1B7F79` | First-principles (DFT) stage; the single feedback loop | `#D5ECEA` |
| ORANGE | `#E08200` | A problem / what the data showed | `#FCE8C8` |
| GREEN | `#2E7D32` | Outcome, certification, "why it is better" | `#D3ECD4` |
| PURPLE | `#6A4C93` | Caveats, outlook | `#E6E0EF` |
| SUPERSEDED | `#8A96A3` | Superseded method (v1 July) | `#EDF1F6` |
| ARROW | `#5B6672` | Neutral connectors, tree buses | — |

**Colour-vision rule.** Some role pairs are close for colour-blind readers: navy/blue, teal/green, navy/purple (`palette_check.py`, 2026-10-09). Role colours therefore always appear **with a text label or numbered badge**. They are never the only distinction between two data series.

## Data encodings (each family passes palette_check on its own)

| Family | Values |
|---|---|
| Sets | training `#C87200` `^`; prediction `#2E6DA4` `o` |
| DFT class | gapped `#008B7A` · narrow `#E0A100` · half-metal `#6A4C93` · metal `#B8C2CC` (stack order gapped, narrow, half-metal, metal) |
| De-emphasised data | `#C9D2DC` |
| Ink | `#1A1A1A` text, `#5B6672` secondary |

Set and class colours never share a panel as competing encodings: set is shown by panel and marker, class by colour.

## Type and geometry

| Token | Value |
|---|---|
| Font | Liberation Sans (Arial-metric; the deck's Segoe UI is not available), pdf.fonttype 42 |
| Sizes (pt) | body 6.5 · small 5.8 · label 7 · heading 7.2 · count 9 · panel letter 9 bold · badge 6.2 · minimum 5.0 (enforced by `glyphs.text` and `qa.check`) |
| Widths | 89 mm (W1), 136 mm (W15), 183 mm (W2); `save(..., width_mm=)` enforces them |
| Line weights (pt) | axis 0.6 · card outline 0.8 · accent 1.2 · arrow 0.9 · hairline 0.45 |
| Card | rounded 1 mm, role tint fill, role outline 0.8 pt, numbered badge radius 2 mm |
| Chevron | solid, 1.1–1.4 mm, coloured like the stage it leaves |
| Loop | one per figure, teal, dashed (3.2, 1.6), labelled once |
| Funnel bar | solid set colour, height 5.4 mm; length ∝ log10(count) when counts span more than 2 decades (state this on the figure); count printed in white bold |
