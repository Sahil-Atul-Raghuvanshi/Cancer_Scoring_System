import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from '@/components/layout/AppShell'
import { useStageCatalog } from '@/features/pipeline/hooks/useStageCatalog'
import { SlideSessionProvider } from '@/features/upload/SlideSession'
import { useUploadCapability } from '@/features/upload/useUploadCapability'
import { DemoPage } from '@/pages/DemoPage'
import { HomePage } from '@/pages/HomePage'

export function App() {
  const { stages, loading } = useStageCatalog()
  const upload = useUploadCapability()

  return (
    <SlideSessionProvider>
      <AppShell>
        <Routes>
          <Route path="/" element={<HomePage stages={stages} />} />
          <Route
            path="/demo"
            element={<DemoPage stages={stages} loading={loading} upload={upload} />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </SlideSessionProvider>
  )
}
