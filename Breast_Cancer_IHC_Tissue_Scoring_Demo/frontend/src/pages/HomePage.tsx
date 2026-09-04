import { Link } from 'react-router-dom'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { padIndex } from '@/lib/format'
import type { PipelineStage } from '@/types/pipeline'

import './home.css'

export function HomePage({ stages }: { stages: PipelineStage[] }) {
  const implemented = stages.filter((stage) => stage.implemented).length
  const trained = stages.filter((stage) => stage.trainsModel).length
  const pretrained = stages.filter((stage) => stage.approach === 'pretrained').length

  return (
    <>
      <section className="container hero">
        <span className="hero__pill">breast · immunohistochemistry · CD44</span>

        <h1 className="hero__title">
          From a glass slide to <em>one defensible number</em>
        </h1>

        <p className="hero__lede">
          A pathology score is a counting problem wrapped in three filtering problems. This
          walkthrough lays out the sixteen steps in the order they must run, and explains why
          that order and no other. Steps 1 to 5 run for real against a slide you upload; the
          rest are documented, not yet built.
        </p>

        <div className="hero__actions">
          <Link to="/demo">
            <Button size="lg" attention>
              Open the walkthrough →
            </Button>
          </Link>
          <a href="#pipeline">
            <Button size="lg" variant="secondary">
              See all sixteen steps
            </Button>
          </a>
        </div>

        <div className="hero__stats">
          {[
            { value: String(stages.length), label: 'pipeline steps' },
            { value: String(implemented), label: 'implemented' },
            { value: String(trained), label: 'models you train' },
            { value: String(pretrained), label: 'models you download' },
          ].map((stat, index) => (
            <div
              key={stat.label}
              className="hero__stat"
              style={{ animationDelay: `${320 + index * 90}ms` }}
            >
              <div className="hero__stat-value">{stat.value}</div>
              <div className="hero__stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="container section" id="pipeline">
        <div className="section__head">
          <h2 className="section__title">The sixteen steps</h2>
          <p className="section__lede">
            Out of sixteen steps, exactly one requires training a model from scratch. Three
            more use weights someone else already trained. The remaining twelve are classical
            image processing and arithmetic — transparent, deterministic, and explainable on
            screen.
          </p>
        </div>

        <div className="steps-grid">
          {stages.map((stage, index) => (
            <article
              key={stage.id}
              className={`step-card${stage.trainsModel ? ' step-card--trained' : ''}${
                stage.implemented ? ' step-card--live' : ''
              }`}
              style={{ animationDelay: `${index * 40}ms` }}
            >
              <div className="step-card__head">
                <span className="step-card__index">{padIndex(stage.index)}</span>
                {stage.implemented ? (
                  <Badge tone="success">live</Badge>
                ) : stage.trainsModel ? (
                  <Badge tone="violet">train this</Badge>
                ) : stage.approach === 'pretrained' ? (
                  <Badge tone="accent">download</Badge>
                ) : (
                  <Badge>classical</Badge>
                )}
              </div>
              <h3 className="step-card__title">{stage.title}</h3>
              <p className="step-card__tagline">{stage.tagline}</p>
            </article>
          ))}
        </div>
      </section>
    </>
  )
}
