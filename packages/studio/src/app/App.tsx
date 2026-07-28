import { useEffect } from 'react';
import { ThemeProvider } from './theme/ThemeProvider';
import { Shell } from './layout/Shell';
import { StructurePane } from './panes/StructurePane';
import { ProcessTreePane } from './panes/ProcessTreePane';
import { PropertiesPane } from './panes/PropertiesPane';
import { ViewportHost } from './viewport/ViewportHost';
import { DropZone } from './import/DropZone';
import { ImportReviewDialog } from './import/ImportReviewDialog';
import { bootstrapSampleModel } from './import/bootstrapSampleModel';

export function App() {
  // First launch: show the bundled sample so the app is never empty.
  useEffect(() => {
    void bootstrapSampleModel();
  }, []);

  return (
    <ThemeProvider>
      <DropZone>
        <Shell
          structure={<StructurePane />}
          viewport={<ViewportHost />}
          processTree={<ProcessTreePane />}
          properties={<PropertiesPane />}
        />
        <ImportReviewDialog />
      </DropZone>
    </ThemeProvider>
  );
}
