import { Link } from "@tanstack/react-router";
import { LINKS } from "@/lib/datasets";

export function SiteFooter() {
  return (
    <footer className="relative z-30 mt-24 overflow-visible border-t border-white/10 bg-black">
      <div className="mx-auto flex w-full max-w-[1560px] flex-col items-center gap-5 overflow-visible px-5 pt-6 pb-14 text-fg sm:h-[92px] sm:flex-row sm:items-center sm:justify-between sm:gap-8 sm:px-7 sm:py-6">
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[13px] text-white/75 sm:text-[15px]">
          <span>© 2026 Meta</span>
          <a href={LINKS.privacy} className="transition-colors hover:text-white">
            Privacy Policy
          </a>
          <a href={LINKS.cookies} className="transition-colors hover:text-white">
            Cookies
          </a>
          <Link to="/credits" className="transition-colors hover:text-white">
            Credits
          </Link>
        </div>
        <a
          href={LINKS.research}
          target="_blank"
          rel="noreferrer"
          className="text-[13px] text-white/75 transition-colors hover:text-white sm:text-[15px]"
        >
          AI.meta.com/research
        </a>
      </div>
    </footer>
  );
}
