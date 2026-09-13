/**
 * Tiny i18n: RU is the default language, EN via the toggle button.
 * Elements carry data-i18n="key"; values are innerHTML strings.
 */

const LANG_KEY = "flora-lang";

export function getLang() {
  try { return localStorage.getItem(LANG_KEY) || "ru"; } catch { return "ru"; }
}
export function setLang(l) {
  try { localStorage.setItem(LANG_KEY, l); } catch {}
}

export const I18N = {
  ru: {
    "m.title": "Flora — ИИ-модель человеческого мозга",
    "m.nav.sub": "Мультимодальная модель кодирования мозга",
    "m.nav.demo": "Демо",
    "m.nav.code": "Код",
    "m.nav.paper": "Статья",
    "m.nav.weights": "Веса",
    "m.drawer.home": "Главная",
    "m.hero.l": "Реальная активность мозга",
    "m.hero.r": "Предсказанная активность мозга",
    "m.hero.h1": "ИИ-модель<br />человеческого мозга",
    "m.hero.sub": "Предсказание нейронных ответов<br />на зрение, звук и язык.",
    "m.cta.demo": "Открыть демо",
    "m.cta.paper": "Читать статью",
    "m.cta.code": "Исходный код",
    "m.cta.model": "Скачать модель",
    "m.s1.title": "Картирование функций мозга: проблема масштаба",
    "m.s1.p1": "Десятилетиями нейробиология упиралась в узкое место: каждый новый эксперимент требует новых записей мозга. Это делает изучение механизмов мозга медленным, дорогим и трудно масштабируемым.",
    "m.s1.p2": "Сегодня мы выпускаем <strong>Flora</strong>. Эта лёгкая модель кодирования — цифровое зеркало корковой активности человека в ответ на зрение, звук и язык: месяцы лабораторной работы превращаются в секунды вычислений, и всё это работает <strong>целиком в вашем браузере</strong>.",
    "m.s2.title": "Flora: трёхстадийная архитектура",
    "m.s2.p": "Flora предсказывает активность мозга по трём стадиям:",
    "m.s2.l1": "<strong>Тримодальное кодирование:</strong> компактные замороженные энкодеры — MobileViT-S (видео, 2&nbsp;fps), Whisper-Tiny (аудио, 16&nbsp;кГц) и MiniLM-L6 (транскрипт) — извлекают признаки стимула, общие для ИИ-моделей и человеческого мозга.",
    "m.s2.l2": "<strong>Mixture-of-Experts фьюжн:</strong> четыре трансформерных слоя с top-2 роутингом, HRF-затухающим вниманием и гейтированным пулингом модальностей обучают универсальное мультимодальное представление при постоянных затратах вычислений.",
    "m.s2.l3": "<strong>Картирование на кору:</strong> кондиционированная на субъекта голова отображает слитое представление на 400 парцелл Schaefer, разворачиваемых на полную поверхность fsaverage5 — <strong>20&nbsp;484 вершины коры</strong>.",
    "m.s3.title": "Лёгкость по дизайну",
    "m.s3.p": "Обычные модели кодирования тратят сотни миллионов параметров на субъекта. Flora замораживает восприятие и обучает лишь фьюжн-стек на <strong>~14M параметров</strong> — стабильно на одной GPU, экспортируется в ONNX и выполняется на GPU вашего ноутбука.",
    "m.s3.c1t": "Высокое качество предсказаний",
    "m.s3.c1p": "Лучший валидационный Pearson r&nbsp;=&nbsp;<strong>0.7278</strong>; по парцеллам r до <strong>0.93</strong> на отложенных клипах — заметно выше линейного ridge-базлайна.",
    "m.s3.c2t": "Обобщение на субъектов",
    "m.s3.c2p": "FiLM-кондиционирование на субъекта поглощает анатомическую вариабельность: новому субъекту нужны лишь два маленьких вектора, без дообучения.",
    "m.s3.c3t": "Инференс в браузере",
    "m.s3.c3p": "ONNX Runtime Web + WebGPU: весь пайплайн — кадры, звук, фьюжн, кора — выполняется локально, без сервера.",
    "m.s4.title": "Динамика обучения",
    "m.s4.p": "Средняя по парцеллам корреляция быстро растёт и достигает пика на <strong>эпохе 52</strong> (r&nbsp;=&nbsp;0.7278), пока train-корреляция продолжает расти — классический признак режима, ограниченного данными. По мере поступления новых фМРТ-записей подход будет улучшаться и дальше.",
    "m.s5.title": "Где модель точнее всего",
    "m.s5.p1": "Точность следует за функцией: зрительная кора лучше всего предсказывается видеопотоком, слуховая и речевая — аудио и текстом. Гейтированный пулинг позволяет каждой области коры взвешивать три потока согласно её роли — ту же топографию можно рассмотреть в демо в режиме <strong>Compare</strong>.",
    "m.s5.p2": "На отложенных клипах с измеренной фМРТ предсказанные корковые «фильмы» Flora совпадают с записанными парцелл за парцеллом — лучшие парцеллы достигают <strong>r&nbsp;≈&nbsp;0.90</strong> по TR.",
    "m.s6.title": "In-silico эксперименты",
    "m.s6.p": "Отключая отдельные потоки модальностей, можно исследовать кору in silico: уберите звук — затихает слуховая кора; уберите транскрипт — гаснут речевые области. Попробуйте вживую — демо перезапускает фьюжн-модель в вашем браузере с любой комбинацией модальностей:",
    "m.chip.video": "Видео",
    "m.chip.audio": "Аудио",
    "m.chip.text": "Текст",
    "m.chip.full": "Полная модель",
    "m.s7.title": "Ускорение будущих исследований",
    "m.s7.p": "Мы открываем статью, код и веса Flora, чтобы ускорить исследования в трёх ключевых направлениях:",
    "m.s7.c1t": "Нейробиология",
    "m.s7.c1p": "Симулируйте корковые ответы, чтобы планировать реальные эксперименты и глубже понимать человеческий мозг.",
    "m.s7.c2t": "Искусственный интеллект",
    "m.s7.c2p": "Ориентируйте развитие архитектур ИИ на эффективность человеческого мозга.",
    "m.s7.c3t": "Здравоохранение",
    "m.s7.c3p": "Заложите основу для улучшения диагностики и лечения заболеваний мозга.",
    "m.s7.more": "Хотите больше?",
    "m.s7.fine": "Статья, код и веса: <a href=\"./Flora-Report.pdf\" target=\"_blank\" rel=\"noreferrer\">статья</a>, <a href=\"https://github.com/qaddasd/Flora\" target=\"_blank\" rel=\"noreferrer\">код</a>, <a href=\"https://github.com/qaddasd/Flora/tree/main/checkpoints\" target=\"_blank\" rel=\"noreferrer\">модель</a>.",
    "m.foot.code": "Код",
    "m.foot.paper": "Статья",
    "m.foot.demo": "Демо",
    "m.banner.p1": "Демо Flora не оптимизировано для маленьких экранов.",
    "m.banner.p2": "Для более комфортного опыта рекомендуем открыть его на большом экране.",
    "m.banner.btn": "Продолжить",
    "m.noscript": "Flora — лёгкая мультимодальная модель кодирования, предсказывающая фМРТ-ответы на естественное видео. Для интерактивной панели нужен JavaScript; для полного демо — браузер с WebGPU.",
    "m.toast": "Статья пока доступна на русском; английская версия — в скором времени.",
    "m.paper.title": "Выберите версию статьи",
    "m.paper.sub": "Flora публикуется в двух версиях — выберите подходящую:",
    "m.paper.off.t": "Официальная версия",
    "m.paper.off.d": "Технический отчёт · 7 страниц · PDF",
    "m.paper.rknp.t": "Версия для РКНП",
    "m.paper.rknp.d": "Научный проект НИШ г. Актау · 20 страниц · PDF",
    "m.paper.open": "Открыть",
    "m.paper.close": "Закрыть",
    "m.paper.note": "Статья пока доступна только на русском языке. Версии на других языках — в скором времени.",

    "d.title": "Flora Демо — живое кодирование мозга в вашем браузере",
    "d.home": "Домой",
    "d.guide": "Показать гид",
    "d.hint.sub": "Выберите клип справа — или загрузите своё видео —<br />и смотрите, как разворачивается предсказанная активность коры.",
    "d.lbl.true": "Истина",
    "d.lbl.pred": "Предсказание",
    "d.seg.true": "Истина",
    "d.seg.compare": "Сравнение",
    "d.seg.pred": "Предсказание",
    "d.seg.normal": "Обычная",
    "d.seg.inflated": "Раздутая",
    "d.seg.open": "Открыть",
    "d.seg.close": "Закрыть",
    "d.tab.browse": "Примеры",
    "d.tab.compare": "Сравнение",
    "d.tab.insilico": "In-Silico",
    "d.tab.about": "О Flora",
    "d.browse.lead": "Сравните предсказания Flora с корковыми ответами на естественное видео. Flora отображает звук и видеоряд клипа на <strong>20&nbsp;484 вершины коры</strong> (fsaverage5) через фьюжн-стек на 14M параметров — <strong>целиком в вашем браузере на WebGPU</strong>.",
    "d.browse.dim": "Шаблонные клипы идут с предсказаниями, посчитанными настоящим PyTorch-пайплайном. Ваши загрузки обрабатываются вживую: кадры → MobileViT-S, звук → Whisper-Tiny, фьюжн → Flora MoE → кора.",
    "d.browse.my": "Мои загрузки",
    "d.browse.templates": "Шаблонные клипы",
    "d.browse.upload": "Загрузить видео",
    "d.browse.upload2": "живой WebGPU-инференс",
    "d.comp.lead": "Отложенные клипы с измеренными корковыми ответами (teacher-модель). Переключайте <strong>Истина</strong> / <strong>Предсказание</strong> / <strong>Сравнение</strong> под мозгом, чтобы рассмотреть согласие.",
    "d.comp.best": "лучший r парцеллы",
    "d.comp.mean": "средний r парцеллы",
    "d.comp.params": "обучаемых параметров",
    "d.is.lead": "Отключайте потоки модальностей и смотрите, как реагирует кора, — эксперимент выполняется <strong>вживую</strong> фьюжн-моделью Flora на WebGPU.",
    "d.is.stim": "Стимул",
    "d.is.mods": "Активные модальности",
    "d.is.video": "Видео",
    "d.is.audio": "Аудио",
    "d.is.text": "Текст",
    "d.is.run": "Запустить эксперимент",
    "d.is.delta": "среднее |Δ| к полному входу",
    "d.about.lead": "<strong>Flora</strong> — лёгкая мультимодальная модель кодирования, предсказывающая фМРТ-ответы на естественное видео.",
    "d.about.w": "Веса: <code>checkpoints/best-epoch=052-val/pearson_r=0.7278.ckpt</code><br />ONNX FP16 экспорт · ONNX Runtime Web · кастомные шейдеры Three.js",
    "d.guide1": "<b>1 · Поверхность коры.</b> fsaverage5, 20 484 вершины. Тяните для вращения, колесо — зум. Горячие точки — предсказанный BOLD-ответ.",
    "d.guide2": "<b>2 · Управление.</b> Истина / Предсказание / Сравнение, морфинг в раздутую поверхность и раскрытие полушарий, как книга.",
    "d.guide3": "<b>3 · Стимулы.</b> Проигрывайте шаблонные клипы, сравнивайте с реальной фМРТ, отключайте модальности in-silico — или загрузите своё видео.",
    "d.guide4": "<b>4 · Шкала активности.</b> Посигнальная нормировка, робастная нормализация по клипу.",
    "d.guide.ok": "Понятно",
    "d.um.title": "Обработка вашего видео",
    "d.um.drop": "Перетащите видео сюда или <u>выберите</u>",
    "d.um.drop2": "MP4 / WebM · анализируется до ~30 с (окно Whisper)",
    "d.ps.decode": "Декодирование видео",
    "d.ps.frames": "5 кадров @ 256²",
    "d.ps.venc": "Энкодер MobileViT-S",
    "d.ps.audio": "Декодирование звука · 16 кГц моно",
    "d.ps.aenc": "Энкодер Whisper-Tiny",
    "d.ps.fusion": "Flora MoE фьюжн → 400 парцелл",
    "d.ps.done": "Спроекцировано на кору",
    "d.um.view": "Показать на мозге",
    "d.noscript": "Демо Flora требует JavaScript и браузера с WebGPU (доступен WASM-фолбэк).",

    "e.title": "404 — Flora",
    "e.text": "Такой страницы нет. Кора, которую вы ищете, не была предсказана.",
    "e.home": "На главную",
    "e.demo": "Открыть демо",
  },

  en: {
    "m.title": "Flora — An AI Model of the Human Brain",
    "m.nav.sub": "Whole-brain encoding model",
    "m.nav.demo": "Demo",
    "m.nav.code": "Code",
    "m.nav.paper": "Paper",
    "m.nav.weights": "Weights",
    "m.drawer.home": "Home",
    "m.hero.l": "Actual brain activity",
    "m.hero.r": "Predicted brain activity",
    "m.hero.h1": "An AI Model of<br />the Human Brain",
    "m.hero.sub": "Predicting neural responses<br />to sight, sound and language.",
    "m.cta.demo": "Explore the Demo",
    "m.cta.paper": "Read the Paper",
    "m.cta.code": "Access the Code",
    "m.cta.model": "Download the Model",
    "m.s1.title": "Mapping Brain Functions: The Challenge of Scale",
    "m.s1.p1": "For decades, neuroscience has faced a major bottleneck: the need for new brain recordings for every new experiment. This has made understanding brain mechanisms slow, costly, and difficult to scale and integrate.",
    "m.s1.p2": "Today, we’re releasing <strong>Flora</strong>. This lightweight encoding model acts as a digital mirror of human cortical activity in response to sight, sound and language — transforming months of lab work into seconds of computation, and running <strong>entirely in your browser</strong>.",
    "m.s2.title": "Flora: a three-stage architecture",
    "m.s2.p": "Flora predicts brain activity through a three-stage pipeline:",
    "m.s2.l1": "<strong>Tri-modal Encoding:</strong> compact frozen encoders — MobileViT-S (video, 2&nbsp;fps), Whisper-Tiny (audio, 16&nbsp;kHz) and MiniLM-L6 (transcript) — capture the stimulus features shared by AI models and the human brain.",
    "m.s2.l2": "<strong>Mixture-of-Experts Fusion:</strong> four transformer layers with top-2 routing, HRF-decay attention and gated modality pooling learn a universal multimodal representation at a constant compute cost.",
    "m.s2.l3": "<strong>Brain Mapping:</strong> a subject-conditioned head maps the fused representation onto 400 Schaefer parcels, unpacked to the full fsaverage5 surface — <strong>20,484 cortical vertices</strong>.",
    "m.s3.title": "Lightweight by Design",
    "m.s3.p": "Conventional encoding models spend hundreds of millions of parameters per subject. Flora keeps the perception frozen and trains only a <strong>~14M-parameter</strong> fusion stack — stable on a single GPU, exportable to ONNX, executable on your laptop’s GPU.",
    "m.s3.c1t": "High-quality predictions",
    "m.s3.c1p": "Best validation Pearson r&nbsp;=&nbsp;<strong>0.7278</strong>; parcel-wise r up to <strong>0.93</strong> on held-out clips — far above a linear ridge baseline.",
    "m.s3.c2t": "Subject generalization",
    "m.s3.c2p": "Per-subject FiLM conditioning absorbs anatomical variability: a new subject needs only two small vectors, no retraining.",
    "m.s3.c3t": "In-browser inference",
    "m.s3.c3p": "ONNX Runtime Web + WebGPU: the full pipeline — frames, audio, fusion, cortex — runs locally with no server.",
    "m.s4.title": "Training Dynamics",
    "m.s4.p": "Mean per-parcel correlation climbs steeply and peaks at <strong>epoch 52</strong> (r&nbsp;=&nbsp;0.7278), while train correlation keeps rising — the classic signature of a data-limited regime. As more fMRI recordings become available, this approach is likely to keep improving.",
    "m.s5.title": "Where the Model Is Most Accurate",
    "m.s5.p1": "Accuracy follows function: visual cortex is predicted best by the video stream, auditory and language cortex by audio and text. Gated modality pooling lets each cortical region weigh the three streams according to its role — the same topography you can inspect live in the demo’s <strong>Compare</strong> view.",
    "m.s5.p2": "On held-out clips with measured fMRI, Flora’s predicted cortical movies track the recorded ones parcel by parcel — best parcels reach <strong>r&nbsp;≈&nbsp;0.90</strong> across TRs.",
    "m.s6.title": "In-Silico Experiments",
    "m.s6.p": "By lesioning individual modality streams, we can inspect the cortex in silico: silence the audio and auditory cortex goes quiet; remove the transcript and language areas fade. Try it live — the demo re-runs the fusion model in your browser with any combination of modalities:",
    "m.chip.video": "Video",
    "m.chip.audio": "Audio",
    "m.chip.text": "Text",
    "m.chip.full": "Full model",
    "m.s7.title": "Accelerating Future Research",
    "m.s7.p": "We open-source the Flora paper, code and model weights to help accelerate research across three key areas:",
    "m.s7.c1t": "Neuroscience",
    "m.s7.c1p": "Simulate cortical responses to plan real experiments and deepen our understanding of the human brain.",
    "m.s7.c2t": "Artificial Intelligence",
    "m.s7.c2p": "Guide the development of AI architectures toward the efficiency of the human brain.",
    "m.s7.c3t": "Healthcare",
    "m.s7.c3p": "Provide a foundation to improve the diagnosis and treatment of brain disorders.",
    "m.s7.more": "Want to go further?",
    "m.s7.fine": "Paper, code and weights: <a href=\"./Flora-Report.pdf\" target=\"_blank\" rel=\"noreferrer\">paper</a>, <a href=\"https://github.com/qaddasd/Flora\" target=\"_blank\" rel=\"noreferrer\">code</a>, <a href=\"https://github.com/qaddasd/Flora/tree/main/checkpoints\" target=\"_blank\" rel=\"noreferrer\">model</a>.",
    "m.foot.code": "Code",
    "m.foot.paper": "Paper",
    "m.foot.demo": "Demo",
    "m.banner.p1": "The Flora demo is not optimized for small screens.",
    "m.banner.p2": "For an improved experience, we recommend you try it on a larger screen.",
    "m.banner.btn": "Continue",
    "m.noscript": "Flora — a lightweight multimodal brain encoding model predicting fMRI responses to naturalistic video. This page needs JavaScript for the interactive brain panel; the full demo also requires a WebGPU-capable browser.",
    "m.toast": "The paper is currently available in Russian; English version coming soon.",
    "m.paper.title": "Choose the paper version",
    "m.paper.sub": "Flora is published in two versions — pick the one you need:",
    "m.paper.off.t": "Official version",
    "m.paper.off.d": "Technical report · 7 pages · PDF",
    "m.paper.rknp.t": "RKNP version",
    "m.paper.rknp.d": "NIS Aktau science project · 20 pages · PDF",
    "m.paper.open": "Open",
    "m.paper.close": "Close",
    "m.paper.note": "The paper is currently available in Russian only. Versions in other languages are coming soon.",

    "d.title": "Flora Demo — Live Whole-Brain Encoding in Your Browser",
    "d.home": "Home",
    "d.guide": "Show Guide",
    "d.hint.sub": "Pick a clip on the right — or upload your own video —<br/>and watch predicted cortical activity unfold.",
    "d.lbl.true": "True",
    "d.lbl.pred": "Predicted",
    "d.seg.true": "True",
    "d.seg.compare": "Compare",
    "d.seg.pred": "Predicted",
    "d.seg.normal": "Normal",
    "d.seg.inflated": "Inflated",
    "d.seg.open": "Open",
    "d.seg.close": "Close",
    "d.tab.browse": "Browse Examples",
    "d.tab.compare": "Compare Performance",
    "d.tab.insilico": "Explore In-Silico",
    "d.tab.about": "About Flora",
    "d.browse.lead": "Compare Flora's predictions with cortical responses to naturalistic video. Flora maps the soundtrack and visual stream of a clip onto <strong>20,484 cortical vertices</strong> (fsaverage5) through a 14M-parameter mixture-of-experts fusion stack — running <strong>entirely in your browser on WebGPU</strong>.",
    "d.browse.dim": "Template clips below ship with predictions precomputed by the real PyTorch pipeline. Your own uploads are processed live: frames → MobileViT-S, audio → Whisper-Tiny, fusion → Flora MoE → cortex.",
    "d.browse.my": "My uploads",
    "d.browse.templates": "Template clips",
    "d.browse.upload": "Upload your video",
    "d.browse.upload2": "live WebGPU inference",
    "d.comp.lead": "Held-out clips with measured cortical responses (teacher model). Toggle <strong>True</strong>, <strong>Predicted</strong> or <strong>Compare</strong> under the brain to inspect agreement.",
    "d.comp.best": "best parcel r",
    "d.comp.mean": "mean parcel r",
    "d.comp.params": "trainable params",
    "d.is.lead": "Lesion individual modality streams and watch the cortex respond — executed <strong>live</strong> by the Flora fusion model in WebGPU.",
    "d.is.stim": "Stimulus",
    "d.is.mods": "Active modalities",
    "d.is.video": "Video",
    "d.is.audio": "Audio",
    "d.is.text": "Text",
    "d.is.run": "Run experiment",
    "d.is.delta": "mean |Δ| vs full input",
    "d.about.lead": "<strong>Flora</strong> — a lightweight multimodal brain encoding model predicting fMRI responses to naturalistic video.",
    "d.about.w": "Weights: <code>checkpoints/best-epoch=052-val/pearson_r=0.7278.ckpt</code><br/> ONNX FP16 export · ONNX Runtime Web · Three.js custom shaders",
    "d.guide1": "<b>1 · The cortical surface.</b> fsaverage5, 20,484 vertices. Drag to rotate, scroll to zoom. Hot spots = predicted BOLD response.",
    "d.guide2": "<b>2 · Controls.</b> Switch True / Predicted / Compare, morph to the inflated surface, and Open the hemispheres like a book.",
    "d.guide3": "<b>3 · Stimuli.</b> Play template clips, benchmark against real fMRI, lesion modalities in-silico — or upload your own video.",
    "d.guide4": "<b>4 · Activity scale.</b> Vertex-wise signal, robust-normalized per clip.",
    "d.guide.ok": "Got it",
    "d.um.title": "Process your video",
    "d.um.drop": "Drop a video here or <u>browse</u>",
    "d.um.drop2": "MP4 / WebM · up to ~30 s analysed (Whisper window)",
    "d.ps.decode": "Decode video",
    "d.ps.frames": "Sample 5 frames @ 256²",
    "d.ps.venc": "MobileViT-S encoder",
    "d.ps.audio": "Decode audio · 16 kHz mono",
    "d.ps.aenc": "Whisper-Tiny encoder",
    "d.ps.fusion": "Flora MoE fusion → 400 parcels",
    "d.ps.done": "Projected onto cortex",
    "d.um.view": "View on brain",
    "d.noscript": "The Flora demo needs JavaScript and a WebGPU-capable browser (WASM fallback available).",

    "e.title": "404 — Flora",
    "e.text": "This page doesn’t exist. The cortex you’re looking for was never predicted.",
    "e.home": "Back to Home",
    "e.demo": "Open the Demo",
  },
};

export function applyI18n() {
  const lang = getLang();
  const dict = I18N[lang] || I18N.ru;
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = dict[el.dataset.i18n];
    if (v != null) el.innerHTML = v;
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => {
    const v = dict[el.dataset.i18nAria];
    if (v != null) el.setAttribute("aria-label", v);
  });
  const key = document.body?.dataset.titleKey;
  if (key && dict[key]) document.title = dict[key];
  document.querySelectorAll("[data-lang-btn]").forEach((b) => {
    b.textContent = lang === "ru" ? "EN" : "RU";
    b.title = lang === "ru" ? "Switch to English" : "Переключить на русский";
  });
}

export function wireLang() {
  document.querySelectorAll("[data-lang-btn]").forEach((b) =>
    b.addEventListener("click", () => {
      setLang(getLang() === "ru" ? "en" : "ru");
      applyI18n();
      window.dispatchEvent(new CustomEvent("flora:lang"));
    }),
  );
}

export function wirePaperModal() {
  const modal = document.getElementById("paper-modal");
  if (!modal) return;
  const open = () => {
    applyI18n();
    modal.classList.remove("hidden");
    document.body.classList.add("paper-lock");
  };
  const close = () => {
    modal.classList.add("hidden");
    document.body.classList.remove("paper-lock");
  };
  document.addEventListener("click", (e) => {
    if (e.target.closest("#paper-modal [data-paper-close]")) { close(); return; }
    // Direct "Open" buttons inside the modal must open the PDF, not the modal again.
    if (e.target.closest("#paper-modal .paper-opt")) return;
    const trigger = e.target.closest('a[data-paper], a[href$="Flora-Report.pdf"]');
    if (trigger) {
      e.preventDefault();
      document.getElementById("drawer")?.classList.add("hidden");
      open();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });
}

// Backward-compatible alias: previous code wired `wirePaperToast`.
export const wirePaperToast = wirePaperModal;
