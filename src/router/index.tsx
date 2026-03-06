import { createBrowserRouter } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { DeveloperContractsPage } from "@/pages/DeveloperContractsPage";
import { NewTaskPage } from "@/pages/NewTaskPage";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { TaskDetailPage } from "@/pages/TaskDetailPage";
import { TasksPage } from "@/pages/TasksPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: "tasks/new", element: <NewTaskPage /> },
      { path: "tasks", element: <TasksPage /> },
      { path: "tasks/:taskId", element: <TaskDetailPage /> },
      { path: "developer/contracts", element: <DeveloperContractsPage /> },
    ],
  },
  {
    path: "*",
    element: <NotFoundPage />,
  },
]);
