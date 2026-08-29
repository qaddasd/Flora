import { create } from "zustand";
import {
  type ColorMode,
  type DemoMode,
  type SurfaceMode,
  NATURALISTIC,
} from "./datasets";

type DemoState = {
  mode: DemoMode;
  colorMode: ColorMode;
  surface: SurfaceMode;
  brainOpen: boolean;
  exampleId: string;
  staticId: string;
  expanded: boolean;
  menuOpen: boolean;
  loading: boolean;
  time: number;
  duration: number;
  playing: boolean;
  muted: boolean;
  rate: number;
  termsAccepted: boolean;
  cookiesAccepted: boolean;
  showCookies: boolean;
  showTerms: boolean;
  setMode: (mode: DemoMode) => void;
  setColorMode: (colorMode: ColorMode) => void;
  setSurface: (surface: SurfaceMode) => void;
  setBrainOpen: (brainOpen: boolean) => void;
  setExampleId: (exampleId: string) => void;
  setStaticId: (staticId: string) => void;
  setExpanded: (expanded: boolean) => void;
  setMenuOpen: (menuOpen: boolean) => void;
  setLoading: (loading: boolean) => void;
  setTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setPlaying: (playing: boolean) => void;
  setMuted: (muted: boolean) => void;
  cycleRate: () => void;
  acceptCookies: () => void;
  acceptTerms: () => void;
  openDemo: () => void;
};

const COOKIE_KEY = "tribe_cookie_consent";
const TERMS_KEY = "tribe_terms_accepted";

function readFlag(key: string) {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

export const useDemoStore = create<DemoState>((set, get) => ({
  mode: "naturalistic",
  colorMode: "compare",
  surface: "normal",
  brainOpen: false,
  exampleId: NATURALISTIC[0]?.id ?? "",
  staticId: "places",
  expanded: false,
  menuOpen: false,
  loading: true,
  time: 0,
  duration: 0,
  playing: true,
  muted: true,
  rate: 1,
  termsAccepted: false,
  cookiesAccepted: false,
  showCookies: false,
  showTerms: false,
  setMode: (mode) =>
    set({
      mode,
      menuOpen: false,
      brainOpen: mode === "naturalistic" ? get().brainOpen : false,
      staticId:
        mode === "insilico"
          ? "places"
          : mode === "performance"
            ? "trainVideo"
            : mode === "rgb"
              ? "rgball"
              : get().staticId,
      colorMode: mode === "naturalistic" || mode === "insilico" ? get().colorMode : "predicted",
    }),
  setColorMode: (colorMode) =>
    set({ colorMode, brainOpen: colorMode === "compare" ? false : get().brainOpen }),
  setSurface: (surface) => set({ surface }),
  setBrainOpen: (brainOpen) => set({ brainOpen }),
  setExampleId: (exampleId) => set({ exampleId, time: 0 }),
  setStaticId: (staticId) => set({ staticId }),
  setExpanded: (expanded) => set({ expanded, menuOpen: false }),
  setMenuOpen: (menuOpen) => set({ menuOpen }),
  setLoading: (loading) => set({ loading }),
  setTime: (time) => set({ time }),
  setDuration: (duration) => set({ duration }),
  setPlaying: (playing) => set({ playing }),
  setMuted: (muted) => set({ muted }),
  cycleRate: () => set({ rate: get().rate >= 2 ? 1 : 2 }),
  acceptCookies: () => {
    try {
      window.localStorage.setItem(COOKIE_KEY, "1");
    } catch {
      /* ignore */
    }
    set({ cookiesAccepted: true, showCookies: false });
  },
  acceptTerms: () => {
    try {
      window.localStorage.setItem(TERMS_KEY, "1");
    } catch {
      /* ignore */
    }
    set({ termsAccepted: true, showTerms: false });
  },
  openDemo: () => {
    if (!get().termsAccepted) {
      set({ showTerms: true });
      return;
    }
    if (typeof window !== "undefined" && window.innerWidth <= 1024) {
      window.dispatchEvent(new CustomEvent("demo-unavailable"));
      return;
    }
    set({ expanded: true });
  },
}));

export function hydrateConsent() {
  const cookies = readFlag(COOKIE_KEY);
  const terms = readFlag(TERMS_KEY);
  useDemoStore.setState({
    cookiesAccepted: cookies,
    termsAccepted: terms,
    showCookies: !cookies,
  });
}
