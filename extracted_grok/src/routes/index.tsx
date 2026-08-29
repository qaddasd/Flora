import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Article } from "@/components/article";
import { DemoPanel, ExpandedDemo } from "@/components/demo-panel";
import { Hero } from "@/components/hero";
import { CookieModal, MobileBanner, TermsModal } from "@/components/modals";
import { NavBar } from "@/components/nav-bar";
import { SiteFooter } from "@/components/site-footer";
import { hydrateConsent, useDemoStore } from "@/lib/demo-store";
import { SECTION_TO_MODE } from "@/lib/content";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  const [navSolid, setNavSolid] = useState(false);
  const setMode = useDemoStore((s) => s.setMode);
  const expanded = useDemoStore((s) => s.expanded);

  useEffect(() => {
    hydrateConsent();
  }, []);

  useEffect(() => {
    const onScroll = () => {
      const hero = document.getElementById("hero");
      const h = hero?.clientHeight ?? 0;
      setNavSolid(window.scrollY > h * 0.6);
      if (useDemoStore.getState().expanded) return;
      const y = window.innerHeight * 0.4;
      let current = "intro";
      for (const id of Object.keys(SECTION_TO_MODE)) {
        const el = document.getElementById(id);
        if (!el) continue;
        if (el.getBoundingClientRect().top <= y) current = id;
      }
      const mode = SECTION_TO_MODE[current];
      if (mode && useDemoStore.getState().mode !== mode) setMode(mode);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [setMode]);

  useEffect(() => {
    document.documentElement.style.overflow = expanded ? "hidden" : "";
    return () => {
      document.documentElement.style.overflow = "";
    };
  }, [expanded]);

  return (
    <div className="min-h-screen bg-black text-fg">
      <NavBar solid={navSolid} />
      <Hero />
      <main className="relative mx-auto w-full max-w-[1500px] px-4 pb-24 lg:px-8">
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-12 lg:gap-12">
          <div className="lg:col-span-7">
            <Article />
          </div>
          <div className="hidden lg:col-span-5 lg:block">
            {!expanded ? (
              <div className="sticky top-[96px] h-[min(72vh,640px)]">
                <DemoPanel />
              </div>
            ) : (
              <div className="sticky top-[96px] h-[min(72vh,640px)]" />
            )}
          </div>
        </div>
      </main>
      <SiteFooter />
      <CookieModal />
      <TermsModal />
      <ExpandedDemo />
      <MobileBanner />
    </div>
  );
}
