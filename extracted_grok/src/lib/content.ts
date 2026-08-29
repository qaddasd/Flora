import { LINKS } from "./datasets";

export type ArticleSection = {
  id: string;
  title: string;
  body: string;
};

export const SECTIONS: ArticleSection[] = [
  {
    id: "intro",
    title: "Mapping Brain Functions: The Challenge of Scale",
    body: `For decades, neuroscience has faced a major bottleneck: the need for new brain recordings for every new experiment. This has made understanding brain mechanisms slow, costly, and difficult to scale and integrate.

Today, we’re releasing TRIBE v2. This foundation model acts as a digital mirror of human brain activity in response to sight, sound and language – transforming months of lab work into seconds of computation.`,
  },
  {
    id: "features",
    title: "TRIBE v2: a three-stage architecture",
    body: `TRIBE v2 predicts brain activity through a three-stage pipeline:

1. **Tri-modal Encoding:** The model utilizes pretrained [audio](${LINKS.audio}), [video](${LINKS.video}) and [text](${LINKS.text}) embeddings to capture the features shared by AI models and the human brain.

2. **Universal Integration:** These embeddings are processed by a transformer that can learn universal representations shared across all stimuli, tasks, and individuals.

3. **Brain Mapping:** Finally, a subject layer maps these universal representations onto individual fMRI voxels – 3D pixels that track neural activity through slow changes in blood flow and oxygenation.

[FIGURE id=figure1]

[FIGURE id=figure1Animation]`,
  },
  {
    id: "performance",
    title: "Key Improvements",
    body: `Building on [TRIBE](${LINKS.tribeV1}), our [Algonauts 2025](${LINKS.algonauts})-winning architecture, we scaled TRIBE v2 across three key dimensions:

[CARDS id=improvements]

These improvements evolve TRIBE from a competition-winning model to a practical tool for neuroscience research and scientific simulation.`,
  },
  {
    id: "generalization",
    title: "Scaling Laws",
    body: `TRIBE v2 follows a scaling law: performance increases log-linearly as we train it with more data. Because this prediction performance has not yet reached a plateau, this approach will likely continue to improve as more fMRI data become available.

[FIGURE id=figure2]`,
  },
  {
    id: "vision",
    title: "Outperforming individual fMRI scans",
    body: `Unlike models built for a single person, TRIBE v2 identifies how patterns of neural activity are shared across individuals. This enables zero-shot generalization: TRIBE v2 accurately predicts the brain response to new stimuli, tasks and subjects without retraining—achieving a 2-3x improvement over standard methods on auditory and visual datasets.

Surprisingly, **TRIBE v2’s predictions are often more representative of the typical response than an actual fMRI scan.** While raw recordings are inherently noisy – distorted by heartbeats, movement and device artifacts – TRIBE v2 predicts a canonical brain response, which is actually more correlated with the group’s average neural activity than almost any single fMRI recording.

[FIGURE id=figure3]`,
  },
  {
    id: "language",
    title: "In-silico experiments replicate seminal discoveries",
    body: `By simulating classic experimental protocols, we can inspect the brain in silico. For example, when presented with a sentence, TRIBE v2 predicts fMRI activity in the well known language network. This approach can effectively identify the brain areas typically associated with:

[CHIPS id=features]`,
  },
  {
    id: "multimodal",
    title: "How the brain builds an integrated representation of the world",
    body: `Finally, TRIBE v2 offers a powerful tool to understand how and where the auditory, visual and linguistic representations are integrated in the brain: Ablation experiments show where each pretrained embedding specifically predict brain activity across both group averages and individuals.`,
  },
  {
    id: "explore",
    title: "Accelerating Future Research",
    body: `As per usual, we open-source the TRIBE v2 paper, code and model weights to help accelerate research across three key areas:

[CARDS id=research]

Want to go further?`,
  },
];

export const IMPROVEMENT_CARDS = [
  {
    id: "voxels",
    label: "Scaled resolution",
    icon: "/assets/magnifying-glass-frame-f50e581b.svg",
    description:
      "TRIBE v2 predicts whole-brain activity across 70,000 voxels—a far more precise picture than the 1,000 cortical predictions of TRIBE v1.",
  },
  {
    id: "participants",
    label: "Scaled data",
    icon: "/assets/three-people-overlapping-50208b54.svg",
    description:
      "Where TRIBE v1 trained on 4 volunteers, TRIBE v2 combines long-form recordings with large cohorts to enable zero-shot generalization.",
  },
  {
    id: "accuracy",
    label: "High-quality predictions",
    icon: "/assets/chart-bar-891e2d35.svg",
    description:
      "TRIBE v2 substantially outperforms traditional baseline models, delivering high-fidelity predictions for complex stimuli.",
  },
];

export const RESEARCH_CARDS = [
  {
    id: "neuroscience",
    label: "Neuroscience",
    icon: "/assets/brain-8927c97f.svg",
    description:
      "Simulate brain responses to help plan actual neuroscience experiments and ultimately deepen our understanding of the human brain.",
  },
  {
    id: "ai",
    label: "Artificial Intelligence",
    icon: "/assets/ai-afe79212.svg",
    description:
      "Guide the development of AI architectures to match the efficiency of the human brain.",
  },
  {
    id: "healthcare",
    label: "Healthcare",
    icon: "/assets/cross-briefcase-69a2f9c0.svg",
    description: "Provide a foundation to improve the diagnosis and treatment of brain disorders.",
  },
];

export const FEATURE_CHIPS = [
  {
    id: "places",
    label: "Places",
    icon: "/assets/figure1-008f21a8.svg",
    staticId: "places",
    tooltip:
      "Replicating the Parahippocampal Place Area (PPA): When shown a landscape or room, TRIBE triggers the same region used for spatial navigation and environmental awareness that fMRI studies established in the 1990s.",
  },
  {
    id: "bodies",
    label: "Bodies",
    icon: "/assets/three-people-overlapping-50208b54.svg",
    staticId: "bodies",
    tooltip:
      "Replicating the Extrastriate Body Area (EBA): When viewing human forms, TRIBE engages the same region identified as a specialized hub for body perception and movement.",
  },
  {
    id: "faces",
    label: "Faces",
    icon: "/assets/theater-mask-stack-ea7d9a10.svg",
    staticId: "faces",
    tooltip:
      "Replicating the Fusiform Face Area (FFA): TRIBE activates the same face-selective region that neuroscientists identified decades ago as the brain’s dedicated center for facial recognition.",
  },
  {
    id: "syntax",
    label: "Speech",
    icon: "/assets/speech-24d1813a.svg",
    staticId: "talkNoTalk",
    tooltip:
      "Recovering the regions involved in language, such as the Superior Temporal Sulcus and Broca's area.",
  },
  {
    id: "word",
    label: "Semantics",
    icon: "/assets/lightbulb-7136144c.svg",
    staticId: "sentenceWord",
    tooltip:
      "Recovering the high-level regions involved in language comprehension, such as Wernicke's area.",
  },
  {
    id: "emotions",
    label: "Emotions",
    icon: "/assets/theater-mask-stack-ea7d9a10.svg",
    staticId: "emotionalPhysicalPain",
    tooltip: "Replicating affective responses across the brain.",
  },
];

export const FIGURES: Record<string, { src: string; alt: string }> = {
  figure1: { src: "/assets/figure1-008f21a8.svg", alt: "Figure 1 — three-stage architecture" },
  figure1Animation: {
    src: "/images/figure1-animation.gif",
    alt: "Figure 1 animation",
  },
  figure2: { src: "/assets/figure2-1c314f0e.svg", alt: "Figure 2 — scaling laws" },
  figure3: { src: "/assets/figure3-8faf625b.svg", alt: "Figure 3 — outperforming individual scans" },
};

export const MODE_COPY: Record<
  string,
  { paragraphs: string[] }
> = {
  naturalistic: {
    paragraphs: [
      "Compare TRIBEv2’s predictions with real brain scans from people watching media clips.",
      "Real scans capture the brain’s full state – including scanner noise, wandering thoughts, and natural variability between individuals – while TRIBE v2 is trained to predict isolated responses to only what is seen and heard. This is the first model to predict whole-brain responses at this resolution for people it has never interacted with.",
    ],
  },
  performance: {
    paragraphs: [
      "This view maps the correlation between TRIBE v2's predictions and actual brain recordings. The highest correlations are in the regions expectedly engaged by the task. For example, movie matching peak in the occipital cortex (back of the brain), which is associated with vision.",
      "Weaker correlations can reflect error caused by a variety of factors including individual variability and scanner noise from raw fMRI scans.",
      "On average, TRIBE v2 achieves 50% higher correlations than traditional models.",
    ],
  },
  insilico: {
    paragraphs: [
      "Select a feature — such as faces, places, or emotions — to see which brain areas the model predicts will respond. These predictions are generated entirely computationally, with no brain scanner or participant required.",
      "These predictions replicate well-established findings from decades of physical neuroscience research. This is the first known computer-simulated replication of classic neuroscientific results, turning months of lab work into seconds of computation.",
    ],
  },
  rgb: {
    paragraphs: [
      "See which brain areas rely most on text, audio, or video information.",
      "This multimodal breakdown is uniquely possible because TRIBE v2 integrates all three input streams simultaneously — reflecting how the brain itself combines information across senses. The subject selector shows how this topography varies across participants.",
    ],
  },
};

export const SECTION_TO_MODE: Record<string, "naturalistic" | "performance" | "insilico" | "rgb"> =
  {
    intro: "naturalistic",
    features: "naturalistic",
    performance: "performance",
    generalization: "performance",
    vision: "performance",
    language: "insilico",
    multimodal: "rgb",
    explore: "rgb",
  };
