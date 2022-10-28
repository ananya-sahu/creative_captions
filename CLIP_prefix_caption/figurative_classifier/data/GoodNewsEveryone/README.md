
# Good News Everyone:  A Corpus of News Headlines Annotated with Emotions, Semantic Roles and Reader Perception

This `.zip` file contains the GoodNewsEveryone! dataset. It contains the
adjudicated and raw annotations. Note that only the annotations of the dataset
are licensed under the Creative Commons Attribution 4.0 International license,
see the file ``LICENSE-annotations``.

This zip file contains the following files:

* `README.md`: this file
* `LICENSE-annotations`: the license of our annotations
    * Note that this license applies *only* to the annotations. The original
      headlines are the property of the original publishers as indicated in the
      corpus file.
* `task1.pdf`: questionaire used for the first annotation task as described in the paper
* `task2.pdf`: questionaire used for the second annotation task as described in the paper
* `gne-release-v1.0.jsonl`: main corpus file
* `gne-release-v1.0.tsv`: a `tsv` version of our `gold` annotations

## Reference

If you use this dataset, please cite:

```
  @inprocedings{Bostan2020,
      author = {Laura Bostan, Evgeny Kim, Roman Klinger},
      title = {Good News Everyone: A Corpus of News Headlines Annotated with \\ Emotions, Semantic Roles and Reader Perception},
      booktitle = {Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)},
      year = {2020},
      month = {may},
      date = {11-16},
      language = {english},
      location = {Marseille, France},
      note = {preprint available at \url{https://arxiv.org/abs/1912.03184}},
  }
```

## Content
The file `gne-release-v1.0.jsonl` contains the annotated dataset.

Each line is a JSON encoded document, representing a single headline and its
annotations. The top level contains a unique ``id``, the ``headline`` itself
and its ``url``, an object containing ``meta`` information (such as the
``source``, the ``country``, and the ``bias`` according to the [Media Bias
Chart](https://www.adfontesmedia.com/product/media-bias-chart-5-0-downloadable-image-and-standard-license/?v=402f03a963ba)
when given), as well as the most important part: the ``annotations`` object.
Per type of annotation, this contains an object again, e.g. ``most_dominant``
for the annotation of the most dominant emotion, ``other_emotions``,
``reader_emotions``, ``intensity``, ``cue``, ``experiencer``, ``cause`` and
``target``.

These objects contain the key ``gold`` for the adjudicated annotation (and the
key ``raw`` for all the annotations), which is either a single atom, or a list
of annotations deemed correct.

The process of the adjudication to the gold annotation is described in the
paper.

### Example:

~~~json
{
"headline": "Dan Crenshaw slams Chuck Schumer for ‘immature and deeply cynical’ reaction to the deal with Mexico",
"meta": {
    "phase1_rank": 4,
    "source": "Twitchy",
    "country": "US",
    "bias": {
      "vertical": "14",
      "horizontal": "29"
    },
"annotations": {
    "dominant_emotion": {
      "gold": "anger"
    },
    "other_emotions": {
      "gold": "annoyance"
    },
    "reader_emotions": {
      "gold": "annoyance"
    },
    "intensity": {
      "gold": "medium"
    },
    "cause": {
      "gold": [
        [
          "‘immature and deeply cynical’ reaction to the deal with mexico"
        ]
      ]
    },
    "cue": {
      "gold": [
        [
          "slams"
        ]
      ]
    "experiencer": {
      "gold": [
        [
          "dan crenshaw"
        ]
      ]
    },
    "target": {
      "gold": [
        [
          "chuck schumer"
        ]
      ]
    },
   }
 }
~~~

----


## Tip

Use [`jq`](https://stedolan.github.io/jq/manual/) for an easy interaction with the corpus.

Examples of how to use it for various tasks:

- page through the corpus interactively
`jq . <gne-release-v1.0.jsonl | less `

- count how often instances are annotated with the stimulus role
`jq <gne-release-v1.0.jsonl 'select(.annotations.cause.gold != [[]]) | .id' | wc -l`

- get the gold annotations in a `tsv` format
`jq -cr '[.id,.headline,.url, .meta.bias.horizontal, .meta.bias.vertical, .meta.country, .meta.source, .annotations.dominant_emotion.gold, .annotations.intensity.gold, (select(.annotations.cause.gold != [[]]) | .annotations.cause.gold | tostring),  (select(.annotations.experiencer.gold != [[]]) | .annotations.experiencer.gold | tostring),  (select(.annotations.target.gold != [[]]) | .annotations.target.gold | tostring), (select(.annotations.cue.gold != [[]]) | .annotations.cue.gold | tostring),  (select(.annotations.other_emotions.gold != [[]]) | .annotations.other_emotions.gold | tostring),  (select(.annotations.reader_emotions.gold != [[]]) | .annotations.reader_emotions.gold | tostring) ] | @tsv' < gne-release-v1.0.jsonl > gne-release-v1.0.tsv`


## Info

For up to date information check out the page of our corpus:
<https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/goodnewseveryone/>


## Contact
Laura Ana Maria Bostan: laura.bostan@ims.uni-stuttgart.de
Evgeny Kim: evgeny.kim@ims.uni-stuttgart.de
Roman Klinger: roman.klinger@ims.uni-stuttgart.de


