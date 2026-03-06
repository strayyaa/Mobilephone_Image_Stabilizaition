import { create } from "zustand";
import { createTask, getTaskDetail, getTasks } from "@/mock/api";
import type { CreateTaskRequest, TaskDetail, TaskQuery, TaskSummary } from "@/types/contracts";

interface TaskState {
  tasks: TaskSummary[];
  selectedTask?: TaskDetail;
  loading: boolean;
  error?: string;
  filters: TaskQuery;
  fetchTasks: (query?: TaskQuery) => Promise<void>;
  fetchTaskDetail: (taskId: string) => Promise<void>;
  createTaskAndRefresh: (payload: CreateTaskRequest) => Promise<TaskSummary>;
  setFilters: (query: TaskQuery) => void;
}

export const useTaskStore = create<TaskState>((set, get) => ({
  tasks: [],
  selectedTask: undefined,
  loading: false,
  error: undefined,
  filters: {},
  setFilters: (query) => set({ filters: query }),
  fetchTasks: async (query) => {
    set({ loading: true, error: undefined });
    try {
      const finalQuery = query ?? get().filters;
      const tasks = await getTasks(finalQuery);
      set({ tasks, filters: finalQuery, loading: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "加载任务失败", loading: false });
    }
  },
  fetchTaskDetail: async (taskId) => {
    set({ loading: true, error: undefined });
    try {
      const detail = await getTaskDetail(taskId);
      set({ selectedTask: detail, loading: false });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "加载详情失败", loading: false });
    }
  },
  createTaskAndRefresh: async (payload) => {
    set({ loading: true, error: undefined });
    try {
      const task = await createTask(payload);
      const tasks = await getTasks(get().filters);
      set({ tasks, loading: false });
      return task;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "创建任务失败", loading: false });
      throw error;
    }
  },
}));
