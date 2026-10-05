/**
 * What step 2 shows before it can run.
 *
 * A fresh clone has no GrandQC checkpoints and no PyTorch, which is the normal
 * state, not an error. So this panel is a checklist rather than a failure
 * message: what is present, what is missing, and the exact command or download
 * that fixes it. Every path shown here comes from the server, so it is the real
 * place it looked - not a guess printed in the client.
 */

import { Badge } from '@/components/ui/Badge'
import type { QCCapability } from '@/types/qc'

import './qc.css'

function Check({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <li className={ok ? 'qc-check qc-check--ok' : 'qc-check qc-check--missing'}>
      <span className="qc-check__mark" aria-hidden>
        {ok ? '✓' : '○'}
      </span>
      <span className="qc-check__body">{children}</span>
    </li>
  )
}

export function QCSetup({ capability }: { capability: QCCapability }) {
  const artefactModels = capability.models.filter((model) => model.role === 'artefact')
  const tissueModel = capability.models.find((model) => model.role === 'tissue')
  const haveAnyArtefact = artefactModels.some((model) => model.found)

  return (
    <div className="qc-setup">
      <div className="qc-setup__head">
        <Badge tone={capability.mode === 'unavailable' ? 'warn' : 'accent'}>
          {capability.mode === 'unavailable' ? 'setup needed' : capability.mode}
        </Badge>
        <p className="qc-setup__reason">{capability.reason}</p>
      </div>

      <ul className="qc-checks">
        <Check ok={capability.featuresAvailable}>
          <strong>Sharpness and contrast measurements</strong>
          <span className="qc-check__hint">
            {capability.featuresAvailable
              ? 'ready — sharpness, contrast and texture can be measured'
              : (capability.featuresProblem ?? 'the scipy package is missing')}
          </span>
        </Check>

        <Check ok={capability.torchInstalled}>
          <strong>PyTorch (runs the models)</strong>
          <span className="qc-check__hint">
            {capability.torchInstalled
              ? `torch ${capability.torchVersion}${
                  capability.smpVersion
                    ? `, segmentation-models-pytorch ${capability.smpVersion}`
                    : ' — segmentation-models-pytorch is missing'
                }`
              : 'needed before either model can run'}
          </span>
        </Check>

        <Check ok={Boolean(tissueModel?.found)}>
          <strong>Tissue model</strong>
          <span className="qc-check__hint">
            {tissueModel?.found
              ? `${tissueModel.name} — finds the tissue in under a second`
              : 'optional. Without it, a simpler colour rule is used, and that rule counts pen marks as tissue'}
          </span>
        </Check>

        <Check ok={haveAnyArtefact}>
          <strong>Problem-area model</strong>
          <span className="qc-check__hint">
            {haveAnyArtefact
              ? artefactModels
                  .filter((model) => model.found)
                  .map((model) => `${model.name} (${model.magnification})`)
                  .join(', ')
              : 'required — this is what finds folds, pen marks, dust, edges and blur'}
          </span>
        </Check>
      </ul>

      {/* --- how to fix it -------------------------------------------------- */}
      {!capability.torchInstalled && (
        <div className="qc-fix">
          <span className="eyebrow">install the missing packages</span>
          <pre className="qc-fix__code mono">
            <code>
              {'cd backend\n'}
              {'.venv\\Scripts\\python -m pip install -r requirements-qc.txt \\\n'}
              {'    --extra-index-url https://download.pytorch.org/whl/cpu'}
            </code>
          </pre>
        </div>
      )}

      {capability.downloads.length > 0 && !capability.ready && (
        <div className="qc-fix">
          <span className="eyebrow">download the model files</span>
          <ul className="qc-downloads">
            {capability.downloads.map((download) => (
              <li key={download.url}>
                <a href={download.url} target="_blank" rel="noreferrer">
                  {download.label}
                </a>
                <span className="qc-downloads__into mono">→ {download.targetDir}</span>
                <span className="qc-downloads__files mono">{download.files.join(' · ')}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <details className="qc-paths">
        <summary>Where we looked for the model files</summary>
        <ul className="mono">
          {capability.searchedPaths.map((path) => (
            <li key={path} className={path === capability.modelsRoot ? 'qc-paths__hit' : undefined}>
              {path}
              {path === capability.modelsRoot && ' ← using this one'}
            </li>
          ))}
        </ul>
        <p className="qc-paths__note">
          To use a different folder, set <code className="mono">QC_MODELS_DIR</code> in{' '}
          <code className="mono">backend/.env</code>. That folder needs a{' '}
          <code className="mono">td/</code> and a <code className="mono">qc/</code> subfolder.
        </p>
      </details>

      <p className="qc-citation">
        {capability.citation}
        <br />
        <em>{capability.licenceNote}</em>
      </p>
    </div>
  )
}
