# Related Work

## Mining User Feedback and App Reviews

Mobile app reviews have long been recognised as a rich source of user opinions,
bug reports, and feature requests, and a substantial body of work in opinion
mining and requirements engineering has focused on extracting structured signal
from this noisy textual data. Recent surveys and tool papers consolidate the
pipeline of collecting, filtering, and analysing user feedback at scale,
highlighting sentiment and topic extraction as core tasks but also noting the
scarcity of fine-grained emotional signal beyond polarity
[1](https://arxiv.org/abs/2407.15519). Dedicated tooling such as AppSent
demonstrates how review corpora can be automatically processed to surface
sentiment-laden opinions about specific app aspects
[2](https://arxiv.org/abs/1907.10191), while transformer-based extractors such
as T-FREX target the complementary problem of identifying app features
mentioned by users, providing the structural backbone on top of which emotional
annotations can be attached [3](https://arxiv.org/abs/2401.03833). Our work
extends this line by moving beyond feature and polarity extraction toward a
structured taxonomy of emotions expressed in app reviews.

## Sentiment and Emotion Classification

Most prior opinion-mining research on reviews frames the task as coarse
polarity classification (positive/negative/neutral) or, at best, as
aspect-based sentiment analysis. In software engineering specifically, Zhang
et al. revisit sentiment analysis in the era of large language models and show
that even modern LLMs struggle with the domain-specific and often ambiguous
affect present in developer- and user-generated text
[4](https://arxiv.org/abs/2310.11113). Beyond polarity, fine-grained emotion
recognition has been pushed forward by resources such as GoEmotions, which
provides 58k Reddit comments labelled with 27 emotion categories derived from
psychological taxonomies [5](https://arxiv.org/abs/2005.00547), and by
distant-supervision approaches like DeepMoji that leverage emoji occurrences
to learn representations useful for emotion, sentiment, and sarcasm detection
[6](https://arxiv.org/abs/1708.00524). These datasets and models mostly target
social media and do not adopt Plutchik's wheel of emotions or account for the
pragmatic peculiarities of app reviews (mixed praise/complaint, feature
requests, star-rating bias), a gap our annotation framework addresses.

## Pretrained Language Models and Transformers

The technical foundation for both classification baselines and modern
annotation assistants rests on pretrained transformer encoders. BERT
established the pretrain-then-fine-tune paradigm that underpins most
state-of-the-art text classifiers, including the models routinely fine-tuned
on GoEmotions and app-review sentiment datasets
[7](https://arxiv.org/abs/1810.04805). More recently, systematic literature
reviews of large language models in software engineering document a rapid
adoption of instruction-tuned LLMs for tasks ranging from code understanding
to review triage, motivating their exploration for emotion annotation as well
[8](https://arxiv.org/abs/2308.10620).

## LLMs as Annotators and Human–LLM Collaboration

A key question tackled in this paper is whether LLMs can replace or assist
human annotators for fine-grained emotion labels. Gilardi et al. report that
ChatGPT can outperform crowd workers on several text-annotation tasks in terms
of agreement and cost, sparking a wave of follow-up studies on the reliability
of LLM labels [9](https://arxiv.org/abs/2303.15056). However, Ashwin et al.
caution that using LLMs for qualitative coding can introduce systematic bias,
particularly for subjective or culturally loaded constructs such as emotions
[10](https://arxiv.org/abs/2309.17147). To mitigate these risks, hybrid
human–LLM annotation systems such as MEGAnno+ have been proposed, coupling
model suggestions with human verification and adjudication workflows
[11](https://arxiv.org/abs/2402.18050). Our evaluation of LLM-based emotion
annotation for app reviews builds directly on this literature, quantifying
inter-annotator agreement between LLMs and humans on a Plutchik-based scheme
and characterising where automation is trustworthy and where human judgement
remains indispensable.
