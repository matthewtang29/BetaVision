# Colour picker: before / after notes

Produced with `tools/eval_colour_pick.py` (each click on its own, default sliders: tolerance 22,
min size 3). "Before" is the code at the first commit (`--impl` on `git show <first commit>:climb_holds.py`), "after" is the
hold-segmenting picker. Side-by-side JPEGs: `eval_out/compare_before_vs_after/` (git-ignored;
regenerate with `--tag before --impl ...`, `--tag after`, `--compare before after`).

**Overall:** the clicked hold is in the result for 40 / 40 clicks (before: 25 / 40) across 6 videos
and 8 frames. Refresh after caching is about the same as before when measured back-to-back
(median 97 ms vs 154 ms for the old code on a busy machine; ~45–75 ms idle).

## The requested clicks (frame total // 2)

| Click | Before | After | Still wrong |
|---|---|---|---|
| 3247 yellow (474,245) | 7, ragged pieces, one hold split in two | 4, clean single outlines | Yellow holds the climber is gripping (mostly hidden by hands) aren't found |
| 3247 pink (245,267) | 7, incl. pieces of magenta holds | 3: clicked + 2 pinks, magenta left alone | The two-tone pink/white hold on the left is not found (its pink part is lighter than the clicked one) |
| 3247 green on gloss (768,949) | 2, read as grey, clicked hold missing | 6, clicked hold whole (highlight included) | The clicked hold touches a second green hold and they come out as one outline |
| 3247 navy (875,235) | 24: wall, trousers, mats | 4: navy + small blue on a volume + the shaded edge of the light-blue hold | That shaded edge (dark blue) is a false piece |
| 3247 white (503,503) | 10 junk, clicked hold missing | 4: clicked + 2 small whites + a white volume | White volume is matched; several other white holds are missed (click them to add) |
| 5468 big yellow (501,533) | 4 | 6 | — |
| 5468 small yellow (416,1643) | 6, different set from the big one | 7, same set as the big-yellow click plus one | — |
| 5468 red on chalk (875,640) | 19 junk, read as grey, clicked hold missing | 4: whole bowl as one outline + 3 reds | Outline is the bowl's convex hull, so it includes a bit of its chalky lip; small reds on the volume are below min size |
| 5468 light blue (405,341) | 10, clicked hold dropped (area cap), shoes / shorts | 5: clicked macro + blue volume face + blue hold | Two light-blue streaks on the big blue volume are matched; the clicked outline has a notch where it's white-marbled |
| 5468 green (213,171) | 8, incl. pieces of yellow holds | 4 greens, no yellow | — |
| 5468 pink (320,32) | 9, incl. pieces of the red bowl | 1 (only the clicked, cut-off hold) | The other pink holds (V9 cluster, small pinks) aren't found: they are a different shade or below min size |

## Other frames and videos (not tuned against first; fixes came from looking at them)

| Click | Before | After | Notes |
|---|---|---|---|
| 3247 f385 yellow / small yellow / pink / magenta / green / navy / white | 6/6/6/6/29/9/5 (white missing) | 6/7/2/4/6/4/4 | green no longer grabs wall; white now includes the clicked hold |
| 5468 f233 yellow / green / light blue / pink / white | 5/23/23/2/3 (light blue, pink, white missing) | 8/11/3/6/10 | white still has a few highlight pieces |
| 3394 red / green / blue | 3/11/18 (blue missing) | 2/10/4 | |
| 3394 "bare wall" 10 px beside a banana-shaped blue hold | 25 junk | 4 (selects that blue hold) | the click is inside the hold's convex hull, so it counts as the hold |
| 5687 pink / yellow / green / red | 11/5/34/0 (pink, red missing) | 7/7/3/10 | |
| 5687 green, click on a heavy chalk cap | 12, read as grey, missing | 4, green, whole hold | |
| 5687 cream (off-white hold) | 6 junk, missing | 9 as a grey hold: clicked + some highlights | other cream holds not found |
| 0302 green big / green / blue | 5/5/7 | 4/4/8 | |
| 3182 yellow / white / pink / light blue | 14/1/4/0 (white, light blue missing) | 10/6/3/3 | |

## What was checked by hand / by script

- Web app (`python -m betavision_app`, built-in browser): AUTO click, a second click on the same
  hold (refused both before and after the first click's outlines came back), bare wall and
  climber clicks (refused, message shown), right-click delete then tolerance 40 / 10 / 22
  (stays deleted, neighbours unaffected), undo of a click and of a delete, a real mouse click on
  the canvas. Handheld IMG_5468 (4 keyframes): clicks, delete on one keyframe carried to the
  others, no console errors.
- OpenCV picker: scripted window events (cv2 GUI calls replaced): click, repeat click, bare
  wall, right-click delete, tolerance changes, Enter; saved info has `colour_models`,
  `deleted_outlines`.
- `--pick`: repeat and bare-wall clicks print why they were skipped; JSON keeps the old keys.
- Saved `*_holds.json` hold entries keep the same keys (`area, bbox, center, contour, id, role,
  source`), which is what `climb_analyze.py --holds` reads.
