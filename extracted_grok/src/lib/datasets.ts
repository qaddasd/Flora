export type DemoMode = "naturalistic" | "performance" | "insilico" | "rgb";
export type ColorMode = "true" | "compare" | "predicted";
export type SurfaceMode = "normal" | "inflated";

export type NaturalisticExample = {
  kind: "dynamic";
  id: string;
  title: string;
  subtitle: string;
  stimulusSrc: string;
  brainColorsUrl: string;
  thumbSrc: string;
};

export const MODES: {
  id: DemoMode;
  label: string;
  action: string;
  tooltip: string;
}[] = [
  {
    id: "naturalistic",
    label: "Examples",
    action: "Browse Examples",
    tooltip: "True and predicted brain responses to excerpts from media.",
  },
  {
    id: "performance",
    label: "Performance",
    action: "Compare Performance",
    tooltip: "Shows the correlation between true and predicted brain responses.",
  },
  {
    id: "insilico",
    label: "In-Silico",
    action: "Explore In-Silico",
    tooltip:
      "Shows TRIBE's prediction of which areas of the brain are responsive to specific features.",
  },
  {
    id: "rgb",
    label: "Multimodality",
    action: "Learn about Multimodality",
    tooltip:
      "Shows how well each brain area is predicted by text, audio and video features.",
  },
];

export const NATURALISTIC: NaturalisticExample[] = [
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start750",
    title: "Vanessen 2023 timeline 0 start 750",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start400",
    title: "Vanessen 2023 timeline 0 start 400",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-400/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-400/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-400/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start50",
    title: "Vanessen 2023 timeline 0 start 50",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-50/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-50/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-50/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start100",
    title: "Vanessen 2023 timeline 0 start 100",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-100/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-100/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-100/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start300",
    title: "Vanessen 2023 timeline 0 start 300",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-300/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-300/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-300/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start200",
    title: "Vanessen 2023 timeline 0 start 200",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-200/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-200/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-200/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start350",
    title: "Vanessen 2023 timeline 0 start 350",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-350/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-350/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-350/thumbnail.jpg",
  },
  {
    kind: "dynamic",
    id: "vanessen2023timeline0start600",
    title: "Vanessen 2023 timeline 0 start 600",
    subtitle: "Demo clip",
    stimulusSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-600/stim.mp4",
    brainColorsUrl:
      "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-600/face-colors.zip",
    thumbSrc: "/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-600/thumbnail.jpg",
  },
];

export const INSILICO = [
  { id: "places", title: "Places", url: "/data/static/in-silico/places-others/face-colors.zip" },
  { id: "bodies", title: "Bodies", url: "/data/static/in-silico/bodies-others/face-colors.zip" },
  { id: "faces", title: "Faces", url: "/data/static/in-silico/faces-others/face-colors.zip" },
  {
    id: "talkNoTalk",
    title: "Talk vs No Talk",
    url: "/data/static/in-silico/talk-no-talk/face-colors.zip",
  },
  {
    id: "sentenceWord",
    title: "Sentence vs Word List",
    url: "/data/static/in-silico/sentence-word/face-colors.zip",
  },
  {
    id: "emotionalPhysicalPain",
    title: "Emotional vs Physical Pain",
    url: "/data/static/in-silico/emotional-physical-pain/face-colors.zip",
  },
] as const;

export const PERFORMANCE = [
  {
    id: "trainVideo",
    title: "Individual responses to movies",
    url: "/data/static/train/tribe-video/face-colors.zip",
  },
  {
    id: "trainSpeech",
    title: "Individual responses to podcasts",
    url: "/data/static/train/tribe-speech/face-colors.zip",
  },
  {
    id: "testSpeech",
    title: "Group responses to podcasts",
    url: "/data/static/test/tribe-speech/face-colors.zip",
  },
  {
    id: "testVideo",
    title: "Group responses to movies",
    url: "/data/static/test/tribe-video/face-colors.zip",
  },
] as const;

export const RGB = [
  { id: "rgball", title: "Average Subject", url: "/data/static/rgb/all/face-colors.zip" },
  { id: "rgb0", title: "Subject 0", url: "/data/static/rgb/0/face-colors.zip" },
  { id: "rgb1", title: "Subject 1", url: "/data/static/rgb/1/face-colors.zip" },
  { id: "rgb2", title: "Subject 2", url: "/data/static/rgb/2/face-colors.zip" },
  { id: "rgb3", title: "Subject 3", url: "/data/static/rgb/3/face-colors.zip" },
] as const;

export const STATIC_URLS: Record<string, string> = {
  ...Object.fromEntries(INSILICO.map((d) => [d.id, d.url])),
  ...Object.fromEntries(PERFORMANCE.map((d) => [d.id, d.url])),
  ...Object.fromEntries(RGB.map((d) => [d.id, d.url])),
  characters: "/data/static/in-silico/characters-others/face-colors.zip",
};

export const LINKS = {
  paper:
    "https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/",
  code: "https://github.com/facebookresearch/tribev2",
  weights: "https://huggingface.co/facebook/tribev2",
  blog: "https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model",
  research: "https://ai.meta.com/research",
  privacy: "https://www.facebook.com/privacy/policy/",
  cookies: "https://www.facebook.com/privacy/policies/cookies/",
  tribeV1: "https://arxiv.org/abs/2507.22229",
  algonauts: "https://algonautsproject.com/2025/archive.html",
  audio: "https://ai.meta.com/research/seamless-communication/",
  video: "https://ai.meta.com/research/vjepa/",
  text: "https://ai.meta.com/blog/meta-llama-3-1/",
};

export const LEGENDS: Record<DemoMode, { src: string; className: string } | null> = {
  naturalistic: { src: "/assets/legend-activity-4d3959ff.svg", className: "h-12" },
  performance: { src: "/assets/legend-encoding-score-fd4aa7c1.svg", className: "h-12" },
  insilico: { src: "/assets/legend-encoding-score-fd4aa7c1.svg", className: "h-12" },
  rgb: { src: "/assets/legend-rgb-18d87b87.svg", className: "h-20" },
};
