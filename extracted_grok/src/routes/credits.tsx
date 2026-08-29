import { createFileRoute, Link } from "@tanstack/react-router";
import { NavBar } from "@/components/nav-bar";
import { SiteFooter } from "@/components/site-footer";
import { LINKS } from "@/lib/datasets";

export const Route = createFileRoute("/credits")({ component: CreditsPage });

function CreditsPage() {
  return (
    <div className="min-h-screen bg-black text-fg">
      <NavBar solid />
      <main className="mx-auto max-w-3xl px-6 pt-28 pb-24">
        <p className="text-sm text-muted">
          <Link to="/" className="hover:text-white">
            ← TRIBE v2
          </Link>
        </p>
        <h1 className="mt-6 font-display text-4xl font-semibold tracking-tight">Credits</h1>
        <div className="mt-8 space-y-6 text-[17px] leading-relaxed text-white/70">
          <p>
            TRIBE v2 is a tri-modal foundation model of vision, audition, and language for in-silico
            neuroscience, developed by Meta FAIR.
          </p>
          <p>
            Brain data from the Individual Brain Charting dataset and related public fMRI corpora used
            to train and evaluate the model.
          </p>
          <p>
            Interactive demo inspired by the public research preview at aidemos.atmeta.com. 3D cortical
            surfaces, stimulus clips, and predicted activity maps are provided for scientific
            illustration.
          </p>
          <ul className="list-disc space-y-2 pl-5">
            <li>
              <a href={LINKS.paper} className="text-white underline underline-offset-4" target="_blank" rel="noreferrer">
                Research paper
              </a>
            </li>
            <li>
              <a href={LINKS.code} className="text-white underline underline-offset-4" target="_blank" rel="noreferrer">
                Code (GitHub)
              </a>
            </li>
            <li>
              <a href={LINKS.weights} className="text-white underline underline-offset-4" target="_blank" rel="noreferrer">
                Model weights (Hugging Face)
              </a>
            </li>
            <li>
              <a href={LINKS.blog} className="text-white underline underline-offset-4" target="_blank" rel="noreferrer">
                Meta AI blog post
              </a>
            </li>
          </ul>
          <p className="text-sm text-white/45">
            This unofficial recreation is for demonstration purposes. TRIBE, Meta, and related marks
            belong to their respective owners.
          </p>
        </div>
      </main>
      <SiteFooter />
    </div>
  );
}
