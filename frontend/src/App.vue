<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "./api";

const videoId = ref("demo-video");
const question = ref("");
const evidence = ref([]);
const videos = ref([]);
const sessionId = ref(null);
const conversation = ref([]);
const result = ref(null);
const loading = ref(false);
const loadingMessage = ref("");
const message = ref("");
const selectedFile = ref(null);
const pendingVideoId = ref(null);
const uploadProgress = ref(0);
const transcriptionTask = ref(null);
const analysisTask = ref(null);
const followUpLoading = ref(false);
const followUpReportId = ref(null);
const followUpProgress = ref(0);
let followUpProgressTimer = null;
const reports = ref([]);
const followUpDrafts = ref({});
const expandedReportId = ref(null);
const reportsSection = ref(null);
const authUser = ref(null);
const authMode = ref("login");
const authUsername = ref("");
const authPassword = ref("");
const authOpen = ref(false);
const authLoading = ref(false);
const authError = ref("");
const authSubmitted = ref(false);
const ingestionMode = ref("file");
const bilibiliBvid = ref("");
const bilibiliPreview = ref(null);
const bilibiliTask = ref(null);
const bilibiliLoading = ref(false);
const authUsernameError = computed(() => {
  const value = authUsername.value.trim();
  if (!value) return "请输入用户名。";
  if (!/^[A-Za-z0-9_.-]{3,32}$/.test(value)) return "用户名需为 3-32 位字母、数字或 . _ -。";
  return "";
});
const authPasswordError = computed(() => {
  if (!authPassword.value) return "请输入密码。";
  if (authPassword.value.length < 8) return "密码至少需要 8 位。";
  return "";
});
const bilibiliProgress = computed(() => {
  const task = bilibiliTask.value;
  if (task && typeof task === "object") {
    const current = Number(task.progress_current ?? task.progressCurrent ?? 0);
    const total = Number(task.progress_total ?? task.progressTotal ?? 0);
    if (total > 0) return Math.min(100, Math.round((current / total) * 100));
    if (task.state === "COMPLETED") return 100;
    if (task.state === "RUNNING") return 24;
    return 12;
  }
  return bilibiliLoading.value ? 8 : 0;
});
const bilibiliProgressLabel = computed(() => {
  const task = bilibiliTask.value;
  if (task && typeof task === "object") {
    if (task.progress_message) return task.progress_message;
    if (task.state === "COMPLETED") return "Bilibili 视频已导入并完成证据整理";
    if (task.state === "FAILED") return task.error || "Bilibili 导入失败";
    return task.state === "QUEUED" ? "正在排队等待导入" : "正在导入 Bilibili 视频";
  }
  return bilibiliLoading.value ? "正在提交 Bilibili 导入任务" : "等待导入";
});
function reportBody(item) {
  return item?.report?.report || item?.report || {};
}
const selectedEvidenceIds = computed(() => new Set(
  (() => {
    const item = reports.value.find((report) => report.id === expandedReportId.value);
    return item ? [
      ...(reportBody(item).evidence || reportBody(item).citations || []),
      ...(item.follow_ups || []).flatMap((followUp) => reportBody(followUp).evidence || reportBody(followUp).citations || []),
    ] : [];
  })().map((item) => Number(item.evidence_id || item.evidenceId || item.id)).filter(Boolean),
));
const currentVideo = computed(() => videos.value.find((item) => item.video_id === videoId.value) || null);
const uploadProgressLabel = computed(() => uploadProgress.value ? `正在上传视频 ${uploadProgress.value}%` : "等待上传");
const transcriptionProgress = computed(() => {
  const task = transcriptionTask.value;
  if (!task) return 0;
  const current = Number(task.progress_current ?? task.progressCurrent ?? 0);
  const total = Number(task.progress_total ?? task.progressTotal ?? 0);
  return total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 42;
});
const transcriptionProgressLabel = computed(() => transcriptionTask.value?.progress_message || "等待转录与 OCR");
const analysisProgress = computed(() => {
  if (analysisTask.value) {
    const current = Number(analysisTask.value.progress_current ?? analysisTask.value.progressCurrent ?? 0);
    const total = Number(analysisTask.value.progress_total ?? analysisTask.value.progressTotal ?? 0);
    return total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 42;
  }
  return followUpLoading.value ? 52 : 0;
});
const followUpProgressLabel = computed(() => "正在生成追问回答");

async function scrollToReports() {
  await nextTick();
  reportsSection.value?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function handleFileChange(event) {
  const file = event.target.files?.[0] || null;
  selectedFile.value = file;
  pendingVideoId.value = file ? `${file.name
      .replace(/\.[^/.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 150) || "uploaded-video"}-${Date.now().toString(36)}` : null;
}
async function loadEvidence() {
  try {
    evidence.value = (await api.listEvidence(videoId.value)).items;
  } catch (error) {
    if (error.status === 401) {
      openAuth();
    } else {
      message.value = error.message;
    }
  }
}

async function loadVideos() {
  try {
    videos.value = (await api.listVideos()).items;
    if (videoId.value === "demo-video" && videos.value.length) {
      videoId.value = videos.value[0].video_id;
    }
  } catch (error) {
    if (error.status === 401) {
      openAuth();
    } else {
      message.value = error.message;
    }
  }
}

async function loadAuthUser() {
  if (!window.localStorage.getItem("tracelens_access_token")) return;
  try {
    authUser.value = await api.me();
  } catch {
    api.logout();
    authUser.value = null;
    authOpen.value = true;
  }
}

async function submitAuth() {
  authSubmitted.value = true;
  authError.value = "";
  if (authUsernameError.value || authPasswordError.value) return;
  authLoading.value = true;
  try {
    const response = authMode.value === "login"
      ? await api.login(authUsername.value, authPassword.value)
      : await api.register(authUsername.value, authPassword.value);
    authUser.value = response.user;
    authOpen.value = false;
    authPassword.value = "";
    authSubmitted.value = false;
    message.value = authMode.value === "login" ? "已登录。" : "账户已创建并登录。";
    await loadVideos();
  } catch (error) {
    if (authMode.value === "login" && error.status === 401) {
      authError.value = "用户名或密码不正确。若尚未注册，请创建账户。";
    } else if (authMode.value === "register" && error.status === 409) {
      authError.value = "该用户名已注册，请直接登录。";
    } else {
      authError.value = error.message || "暂时无法完成登录，请稍后重试。";
    }
  } finally {
    authLoading.value = false;
  }
}

function logout() {
  api.logout().catch(() => {});
  authUser.value = null;
  videos.value = [];
  evidence.value = [];
  reports.value = [];
  message.value = "已退出当前账户。";
}

function toggleAuthMode() {
  authMode.value = authMode.value === "login" ? "register" : "login";
  authError.value = "";
  authSubmitted.value = false;
  authPassword.value = "";
}

function openAuth(mode = "login") {
  authMode.value = mode;
  authError.value = "";
  authSubmitted.value = false;
  authOpen.value = true;
}

function closeAuth() {
  if (!authLoading.value) authOpen.value = false;
}

function selectIngestionMode(mode) {
  ingestionMode.value = mode;
  message.value = "";
  if (mode === "file") {
    bilibiliPreview.value = null;
    bilibiliTask.value = null;
  } else {
    selectedFile.value = null;
    pendingVideoId.value = null;
  }
}

async function previewBilibili() {
  if (!bilibiliBvid.value.trim()) return;
  bilibiliLoading.value = true;
  message.value = "正在读取 Bilibili 视频信息...";
  try {
    bilibiliPreview.value = await api.previewBilibili(bilibiliBvid.value.trim());
  } catch (error) {
    bilibiliPreview.value = null;
    message.value = error.message;
  } finally {
    bilibiliLoading.value = false;
  }
}

async function importBilibili() {
  bilibiliLoading.value = true;
  bilibiliTask.value = null;
  message.value = "正在提交 Bilibili 导入任务...";
  try {
    const payload = await api.importBilibili(bilibiliBvid.value.trim());
    bilibiliTask.value = payload.task_id;
    await waitForBilibiliTask(payload.task_id);
  } catch (error) {
    message.value = error.message;
  } finally {
    bilibiliLoading.value = false;
  }
}

async function waitForBilibiliTask(taskId) {
  const deadline = Date.now() + 30 * 60 * 1000;
  while (Date.now() < deadline) {
    const task = await api.getBilibiliImportStatus(taskId);
    bilibiliTask.value = task;
    if (task.state === "COMPLETED") {
      bilibiliPreview.value = null;
      bilibiliBvid.value = "";
      await loadVideos();
      videoId.value = task.video_id;
      await loadEvidence();
      await loadMemory();
      message.value = "Bilibili 视频已导入，证据时间轴可以使用了。";
      return;
    }
    if (task.state === "FAILED") throw new Error(task.error || "Bilibili 导入失败");
    await new Promise((resolve) => window.setTimeout(resolve, 900));
  }
  throw new Error("Bilibili 导入等待超时，请稍后刷新任务状态。");
}

async function loadMemory() {
  try {
    const memory = await api.getVideoMemory(videoId.value);
    const session = memory.sessions?.[0];
    sessionId.value = session?.session_id || null;
    conversation.value = session?.messages || [];
    reports.value = (await api.listReports(videoId.value)).items || [];
    expandedReportId.value = null;
  } catch (error) {
    if (error.status === 401) {
      openAuth();
    } else {
      message.value = error.message;
    }
  }
}

async function selectVideo(video) {
  videoId.value = video.video_id;
  selectedFile.value = null;
  result.value = null;
  reports.value = [];
  transcriptionTask.value = null;
  analysisTask.value = null;
  await loadEvidence();
  await loadMemory();
}

async function selectCurrentVideo() {
  await selectVideo({ video_id: videoId.value });
}

async function removeVideo(video) {
  if (!window.confirm(`Delete ${video.filename}? This removes the video and its evidence.`)) return;
  loading.value = true;
  uploadProgress.value = 0;
  loadingMessage.value = "Deleting video and evidence...";
  message.value = "";
  try {
    await api.deleteVideo(video.video_id);
    await loadVideos();
    if (videoId.value === video.video_id) {
      videoId.value = "demo-video";
      evidence.value = [];
      result.value = null;
      await loadEvidence();
      sessionId.value = null;
      conversation.value = [];
      reports.value = [];
    }
    message.value = "Video and its evidence were deleted.";
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function seedDemo() {
  loading.value = true;
  loadingMessage.value = "正在准备演示证据...";
  message.value = "";
  try {
    await api.seedDemo(videoId.value);
    await loadEvidence();
    result.value = null;
    message.value = "演示证据已准备好，可以开始提问。";
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function uploadVideo() {
  if (!selectedFile.value) {
    message.value = "请先选择一个视频文件。";
    return;
  }

  loading.value = true;
  uploadProgress.value = 0;
  loadingMessage.value = "正在上传视频...";
  message.value = "";
  try {
    const uploadId = pendingVideoId.value || `uploaded-video-${Date.now().toString(36)}`;
    const payload = await api.uploadVideo(uploadId, selectedFile.value, true, (percent) => {
      uploadProgress.value = percent;
      loadingMessage.value = `正在上传视频 ${percent}%...`;
    });
    videoId.value = payload.video_id;
    pendingVideoId.value = null;
    await loadVideos();
    await loadEvidence();
    result.value = null;
    transcriptionTask.value = payload.transcription_task_id || null;
    if (!transcriptionTask.value) {
      await loadEvidence();
      message.value = `已上传 ${payload.filename}。`;
      uploadProgress.value = 0;
      return;
    }
    await waitForTask(transcriptionTask.value, payload.filename);
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function waitForTask(taskId, filename) {
  const deadline = Date.now() + 30 * 60 * 1000;
  while (Date.now() < deadline) {
    const task = await api.getTask(taskId);
    transcriptionTask.value = task;
    if (task.state === "COMPLETED") {
      await loadEvidence();
      await loadVideos();
      const summary = task.result || {};
      message.value = `已完成 ${filename}，ASR ${summary.asr_count ?? 0} 条，OCR ${summary.ocr_count ?? 0} 条证据已写入。`;
      transcriptionTask.value = null;
      uploadProgress.value = 0;
      return;
    }
    if (task.state === "FAILED") {
      throw new Error(task.error || "视频转录任务失败");
    }
    loadingMessage.value = task.progress_message || (task.state === "RUNNING" ? "正在转录并整理证据..." : "任务排队中...");
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
  throw new Error("视频转录任务超时，请稍后查看视频状态。");
}

async function ask() {
  await submitQuestion(question.value.trim(), null);
}

async function submitQuestion(text, reportSessionId, { preserveScroll = false } = {}) {
  if (!text) return;
  if (transcriptionTask.value && transcriptionTask.value.state !== "COMPLETED") {
    message.value = "视频仍在转录，请等待任务完成后再提问。";
    return;
  }
  loading.value = true;
  loadingMessage.value = "正在检索筛选证据并请求 DeepSeek...";
  message.value = "";
  if (!preserveScroll) await scrollToReports();
  try {
    const response = reportSessionId
      ? await api.ask(text, videoId.value, reportSessionId)
      : await api.createAnalysis(text, videoId.value, sessionId.value);
    if (!reportSessionId && response.taskId && response.state !== "COMPLETED") {
      sessionId.value = response.sessionId || sessionId.value;
      await waitForAnalysisTask(response.taskId);
      return;
    }
    if (!reportSessionId && response.taskId && response.state === "COMPLETED") {
      sessionId.value = response.sessionId || sessionId.value;
      await loadMemory();
      if (reports.value.length) expandedReportId.value = reports.value[0].id;
      await scrollToReports();
      question.value = "";
      return;
    }
    if (!reportSessionId && response.taskId && response.state === "FAILED") {
      result.value = { error: true, answer: response.error || "报告生成失败", provider: "DeepSeek" };
      message.value = response.error || "报告生成失败，请查看后端日志。";
      return;
    }
    if (!response.error || response.report) {
      await loadMemory();
      if (!reportSessionId && reports.value.length) {
        expandedReportId.value = reports.value[0].id;
      }
      if (!reportSessionId) await scrollToReports();
      if (!reportSessionId) question.value = "";
    } else {
      result.value = response;
    }
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function waitForAnalysisTask(taskId) {
  const deadline = Date.now() + 60 * 60 * 1000;
  while (Date.now() < deadline) {
    const task = await api.getAnalysisTask(taskId);
    analysisTask.value = task;
    sessionId.value = task.sessionId || sessionId.value;
    if (task.state === "QUEUED" || task.state === "RUNNING") {
      await scrollToReports();
    }
    if (task.state === "COMPLETED") {
      await loadMemory();
      if (reports.value.length) expandedReportId.value = reports.value[0].id;
      result.value = task.result || null;
      question.value = "";
      analysisTask.value = null;
      await scrollToReports();
      const taskReport = task.result?.report || {};
      const hasCitations = (taskReport.evidence || task.result?.citations || []).length > 0;
      message.value = task.result?.error
        ? "模型请求失败，但本次问题已保存为失败记录。"
        : task.result?.grounded && hasCitations
        ? "报告已生成，已保留筛选后的证据引用。"
        : "分析已完成，但证据不足，未生成可引用的正式报告。";
      return;
    }
    if (task.state === "FAILED") {
      result.value = { error: true, answer: task.error || "报告生成失败", provider: "DeepSeek" };
      message.value = task.error || "报告生成失败，请查看后端日志。";
      analysisTask.value = null;
      return;
    }
    loadingMessage.value = task.message || "正在分析视频并生成报告...";
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  throw new Error("分析任务等待超时，请稍后刷新页面查看状态。");
}

async function submitFollowUp(report) {
  const text = String(followUpDrafts.value[report.id] || "").trim();
  if (!text) return;
  followUpLoading.value = true;
  followUpReportId.value = report.id;
  followUpProgress.value = 8;
  window.clearInterval(followUpProgressTimer);
  followUpProgressTimer = window.setInterval(() => {
    followUpProgress.value = Math.min(88, followUpProgress.value + 7);
  }, 350);
  expandedReportId.value = report.id;
  await nextTick();
  document.getElementById(`follow-up-${report.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  try {
    await submitQuestion(text, report.session_id, { preserveScroll: true });
    expandedReportId.value = report.id;
    followUpDrafts.value = { ...followUpDrafts.value, [report.id]: "" };
    await nextTick();
    const refreshed = reports.value.find((item) => item.id === report.id);
    const latestFollowUp = refreshed?.follow_ups?.at(-1);
    const target = latestFollowUp
      ? document.getElementById(`follow-up-record-${latestFollowUp.id}`)
      : document.getElementById(`report-${report.id}`);
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
  } finally {
    window.clearInterval(followUpProgressTimer);
    followUpProgressTimer = null;
    followUpLoading.value = false;
    followUpReportId.value = null;
    followUpProgress.value = 0;
  }
}

async function removeReport(report) {
  if (!window.confirm("删除这条报告及其追问记录？视频和证据不会被删除。")) return;
  loading.value = true;
  message.value = "";
  try {
    await api.deleteReport(report.id);
    if (expandedReportId.value === report.id) expandedReportId.value = null;
    await loadMemory();
    message.value = "报告及其追问记录已删除。";
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
  }
}

async function removeFollowUp(report, followUp) {
  if (!window.confirm("删除这条追问记录？")) return;
  loading.value = true;
  message.value = "";
  try {
    await api.deleteReport(followUp.id);
    expandedReportId.value = report.id;
    await loadMemory();
    expandedReportId.value = report.id;
    message.value = "追问记录已删除。";
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
  }
}

function restoreReport(item) {
  expandedReportId.value = item.id;
  result.value = item.report?.report || item.report;
  sessionId.value = item.session_id || sessionId.value;
}

function toggleReport(reportId) {
  expandedReportId.value = expandedReportId.value === reportId ? null : reportId;
}

function formatSeconds(seconds) {
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const rest = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function handleAuthExpired() {
  authUser.value = null;
  authOpen.value = true;
  videos.value = [];
  evidence.value = [];
  reports.value = [];
  message.value = "登录已失效，请重新登录。";
}

onMounted(async () => {
  window.addEventListener("tracelens-auth-expired", handleAuthExpired);
  await loadAuthUser();
  await loadVideos();
  await loadEvidence();
  await loadMemory();
});

onBeforeUnmount(() => {
  window.removeEventListener("tracelens-auth-expired", handleAuthExpired);
});
</script>

<!--
<template>
  <main v-if="false" class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">TRACEABLE VIDEO QA</p>
        <h1>TraceLens</h1>
      </div>
      <div class="header-actions">
        <label class="upload-button">
          <input type="file" accept="video/*" @change="handleFileChange" />
          选择视频
        </label>
        <button class="secondary-button" :disabled="loading || !selectedFile" @click="uploadVideo">
          上传并转录
        </button>
        <button class="secondary-button" :disabled="loading" @click="seedDemo">
          准备演示数据
        </button>
      </div>
    </header>

    <section class="workspace">
      <aside v-if="false" class="evidence-panel">
        <div class="panel-heading">
          <div>
            <span class="section-label">证据时间轴</span>
            <h2>{{ evidence.length }} 条证据</h2>
          </div>
          <span
            :class="['status-dot', videoId === 'demo-video' ? 'is-demo' : 'is-uploaded']"
            :title="videoId === 'demo-video' ? '当前为演示数据' : '当前为上传视频数据'"
          ></span>
        </div>

        <label class="field-label" for="video-id">视频标识</label>
        <input id="video-id" v-model="videoId" class="text-input" @change="loadEvidence" />

        <div v-if="videos.length" class="video-library">
          <div class="library-heading">Saved videos</div>
          <div v-for="video in videos" :key="video.video_id" class="video-row">
            <button class="video-select" :class="{ 'is-active': video.video_id === videoId }" @click="selectVideo(video)">
              <strong>{{ video.filename }}</strong>
              <span>{{ video.evidence_count }} evidence items</span>
            </button>
            <button class="delete-button" :disabled="loading" title="Delete video" @click="removeVideo(video)">×</button>
          </div>
        </div>

        <div v-if="evidence.length" class="evidence-list">
          <article v-for="item in evidence" :key="item.id" class="evidence-item">
            <div class="timestamp">
              <span class="evidence-source">{{ item.source || "ASR" }}</span>
              {{ formatSeconds(item.start_seconds) }} - {{ formatSeconds(item.end_seconds) }}
            </div>
            <p>{{ item.text }}</p>
          </article>
        </div>
        <p v-else class="empty-state">还没有证据。请先选择视频并上传或导入视频。</p>
      </aside>

      <section class="answer-panel">
        <div class="panel-heading">
          <div>
            <span class="section-label">证据问答</span>
            <h2>只根据已检索证据回答</h2>
          </div>
          <span
            v-if="result"
            :class="['grounding-badge', result.grounded ? 'is-grounded' : 'is-refused']"
          >
            {{ result.grounded ? "有证据支持" : "证据不足" }}
          </span>
        </div>

        <form class="ask-form" @submit.prevent="ask">
          <textarea
            v-model="question"
            class="question-input"
            rows="4"
            placeholder="例如：这个项目应该如何准备面试？"
          ></textarea>
          <button class="primary-button" :disabled="loading || !question.trim()">
            {{ loading ? "检索中..." : "检索并回答" }}
          </button>
        </form>

        <p v-if="loadingMessage" class="notice progress-notice">{{ loadingMessage }}</p>
        <p v-if="message" class="notice">{{ message }}</p>

        <div v-if="conversation.length" class="conversation-section">
          <div class="conversation-heading">Session memory</div>
          <div v-for="(item, index) in conversation" :key="`${item.created_at}-${index}`" class="conversation-item">
            <span>{{ item.role === "user" ? "You" : "Agent" }}</span>
            <p>{{ item.content }}</p>
          </div>
        </div>

        <div v-if="result" class="result">
          <template v-if="result.kind === 'initial_report' && result.report">
            <article class="report-card">
              <div class="report-kicker">最终报告</div>
              <h3 class="report-title">{{ result.report.title }}</h3>
              <p class="report-answer">{{ result.report.finalAnswer }}</p>
              <section v-if="result.report.conclusions?.length" class="report-section">
                <h4>核心结论</h4>
                <ul>
                  <li v-for="(item, index) in result.report.conclusions" :key="`conclusion-${index}`">{{ item }}</li>
                </ul>
              </section>
              <section v-if="result.report.evidence?.length" class="report-section">
                <div class="report-section-heading">
                  <h4>视频证据</h4>
                  <span>{{ result.report.evidence.length }} 条筛选证据</span>
                </div>
                <div class="report-evidence-list">
                  <article v-for="item in result.report.evidence" :key="item.evidence_id" class="report-evidence">
                    <div class="report-evidence-meta">
                      <strong>E{{ String(item.evidence_id).padStart(3, '0') }}</strong>
                      <span>{{ item.source || 'ASR' }}</span>
                      <span>{{ item.timestamp }}</span>
                    </div>
                    <p>{{ item.text }}</p>
                  </article>
                </div>
              </section>
              <section v-if="result.report.suggestions?.length" class="report-section">
                <h4>后续建议</h4>
                <ul>
                  <li v-for="(item, index) in result.report.suggestions" :key="`suggestion-${index}`">{{ item }}</li>
                </ul>
              </section>
            </article>
          </template>
          <article v-else class="follow-up-answer">
            <div class="report-kicker">追问回答</div>
            <p class="answer">{{ result.answer }}</p>
          </article>

          <div v-if="result.trace && result.trace.length" class="trace-section">
            <details open>
              <summary>Agent 思考过程 ({{ result.trace.length }} 步)</summary>
              <ol class="trace-list">
                <li v-for="(step, index) in result.trace" :key="index" class="trace-step">
                  {{ step }}
                </li>
              </ol>
            </details>
          </div>
          
        </div>
        <div v-else class="result-placeholder">
          <strong>回答会显示在这里</strong>
          <span>每条结论都应该能够回到具体时间段。</span>
        </div>
      </section>
    </section>
  </main>
</template>
-->

<template>
  <main class="shell clean-shell">
    <header class="clean-topbar">
      <div>
        <p class="eyebrow">TRACEABLE VIDEO QA</p>
        <h1>TraceLens</h1>
        <p class="subtitle">从视频证据中生成可追溯的分析报告</p>
      </div>
      <div class="clean-actions">
        <span v-if="authUser" class="account-chip">{{ authUser.username }}</span>
        <button v-if="authUser" class="secondary-button" @click="logout">退出</button>
        <button v-else class="secondary-button" @click="openAuth()">登录 / 注册</button>
      </div>
    </header>

    <div v-if="authOpen" class="auth-backdrop" @click.self="closeAuth">
      <section class="auth-modal" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <header class="auth-modal-header">
          <div><span class="section-label">ACCOUNT</span><h2 id="auth-title">{{ authMode === "login" ? "登录 TraceLens" : "创建账户" }}</h2></div>
          <button type="button" class="icon-button auth-close" aria-label="关闭登录窗口" :disabled="authLoading" @click="closeAuth">×</button>
        </header>
        <div class="auth-mode-tabs" role="tablist" aria-label="账户操作">
          <button type="button" :class="{ 'is-active': authMode === 'login' }" :aria-selected="authMode === 'login'" @click="authMode !== 'login' && toggleAuthMode()">登录</button>
          <button type="button" :class="{ 'is-active': authMode === 'register' }" :aria-selected="authMode === 'register'" @click="authMode !== 'register' && toggleAuthMode()">注册</button>
        </div>
        <form class="auth-modal-form" novalidate @submit.prevent="submitAuth">
          <label for="auth-username">用户名</label>
          <input id="auth-username" v-model.trim="authUsername" class="text-input" :class="{ 'has-error': authSubmitted && authUsernameError }" placeholder="3-32 位字母、数字或 . _ -" autocomplete="username" :disabled="authLoading" @input="authError = ''" />
          <p v-if="authSubmitted && authUsernameError" class="field-error">{{ authUsernameError }}</p>
          <label for="auth-password">密码</label>
          <input id="auth-password" v-model="authPassword" class="text-input" :class="{ 'has-error': authSubmitted && authPasswordError }" type="password" placeholder="至少 8 位" :autocomplete="authMode === 'login' ? 'current-password' : 'new-password'" :disabled="authLoading" @input="authError = ''" />
          <p v-if="authSubmitted && authPasswordError" class="field-error">{{ authPasswordError }}</p>
          <p v-if="authError" class="auth-error" role="alert">{{ authError }}</p>
          <button class="primary-button auth-submit" :disabled="authLoading">{{ authLoading ? "正在处理..." : authMode === "login" ? "登录" : "创建账户" }}</button>
        </form>
        <p class="auth-switch">{{ authMode === "login" ? "还没有账户？" : "已经有账户？" }}<button type="button" class="text-button" :disabled="authLoading" @click="toggleAuthMode">{{ authMode === "login" ? "去注册" : "去登录" }}</button></p>
      </section>
    </div>

    <section class="upload-card">
      <div class="upload-card-heading">
        <div><span class="section-label">VIDEO INGESTION</span><h2>添加一个视频</h2></div>
        <span class="upload-card-caption">上传、转录和 OCR 会在后台持续运行</span>
      </div>
      <div class="ingestion-tabs" role="tablist" aria-label="视频导入方式">
        <button type="button" :class="['ingestion-tab', { 'is-active': ingestionMode === 'file' }]" :aria-selected="ingestionMode === 'file'" @click="selectIngestionMode('file')">本地文件</button>
        <button type="button" :class="['ingestion-tab', { 'is-active': ingestionMode === 'bilibili' }]" :aria-selected="ingestionMode === 'bilibili'" @click="selectIngestionMode('bilibili')">Bilibili BV</button>
      </div>
      <div v-if="ingestionMode === 'file'" class="upload-card-body">
        <label class="upload-dropzone" :class="{ 'has-file': selectedFile }">
          <input type="file" accept="video/*" @change="handleFileChange" />
          <span class="upload-icon">↑</span>
          <span><strong>{{ selectedFile ? selectedFile.name : "选择视频文件" }}</strong><small>{{ selectedFile ? "已准备好上传，点击右侧按钮开始处理" : "支持 MP4、MOV、MKV、WebM 等常见格式" }}</small></span>
        </label>
        <button class="primary-button upload-submit" :disabled="loading || !selectedFile" @click="uploadVideo">{{ loading ? "处理中..." : "上传并转录" }}</button>
      </div>
      <div v-else class="bilibili-import">
        <div class="bilibili-copy"><strong>从 Bilibili 导入</strong><small>输入 BV 号，先预览元数据，再下载并进入同一套转录与证据流程。</small></div>
        <div class="bilibili-controls"><input v-model="bilibiliBvid" class="text-input" placeholder="例如 BV1xx411c7mD" @keyup.enter="previewBilibili" /><button class="secondary-button" :disabled="bilibiliLoading || !bilibiliBvid.trim()" @click="previewBilibili">预览</button></div>
        <div v-if="bilibiliPreview" class="bilibili-preview"><div><strong>{{ bilibiliPreview.title }}</strong><span>{{ bilibiliPreview.uploader || "未知作者" }} · {{ Math.floor((bilibiliPreview.durationSeconds || 0) / 60) }} 分钟</span></div><button class="primary-button" :disabled="bilibiliLoading" @click="importBilibili">确认导入</button></div>
        <div v-if="bilibiliTask || bilibiliLoading" :class="['task-progress-card', 'bilibili-progress-card', { 'is-failed': bilibiliTask?.state === 'FAILED', 'is-complete': bilibiliTask?.state === 'COMPLETED' }]">
          <div class="task-progress-top"><strong>{{ bilibiliProgressLabel }}</strong><span>{{ bilibiliProgress }}%</span></div>
          <div class="progress-track"><span :style="{ width: `${bilibiliProgress}%` }"></span></div>
          <div class="task-steps"><span :class="{ active: bilibiliProgress >= 12 }">下载视频</span><i></i><span :class="{ active: bilibiliProgress >= 50 }">转录 ASR</span><i></i><span :class="{ active: bilibiliProgress >= 75 }">提取 OCR</span><i></i><span :class="{ active: bilibiliProgress >= 100 }">完成</span></div>
        </div>
      </div>
      <div v-if="uploadProgress > 0 && loading" class="task-progress-card upload-progress-card">
        <div class="task-progress-top"><strong>{{ uploadProgressLabel }}</strong><span>{{ uploadProgress }}%</span></div>
        <div class="progress-track"><span :style="{ width: `${uploadProgress}%` }"></span></div>
        <div class="task-steps"><span class="active">上传视频</span></div>
      </div>
      <div v-if="transcriptionTask" class="task-progress-card transcription-progress-card">
        <div class="task-progress-top"><strong>{{ transcriptionProgressLabel }}</strong><span>{{ transcriptionProgress }}%</span></div>
        <div class="progress-track"><span :style="{ width: `${transcriptionProgress}%` }"></span></div>
        <div class="task-steps"><span class="active">转录</span><i></i><span class="active">OCR</span></div>
      </div>
    </section>

    <section class="clean-controls">
      <label for="clean-video-select">当前视频</label>
      <select id="clean-video-select" v-model="videoId" class="clean-video-select" @change="selectCurrentVideo">
        <option v-if="!videos.length" value="demo-video">暂无视频</option>
        <option v-for="video in videos" :key="video.video_id" :value="video.video_id">{{ video.filename }}</option>
      </select>
      <span class="video-state" :class="'is-' + String(currentVideo?.status || 'empty').toLowerCase()">
        {{ currentVideo ? `${currentVideo.evidence_count} 条证据 · ${currentVideo.status}` : "尚未选择视频" }}
      </span>
      <button v-if="currentVideo" class="danger-button" :disabled="loading" @click="removeVideo(currentVideo)">删除视频</button>
      <span class="clean-hint">报告、证据和会话会随当前视频恢复。</span>
    </section>

    <section class="analysis-layout">
      <aside class="evidence-drawer">
        <div class="drawer-heading"><div><span class="section-label">TIMELINE</span><h2>证据时间轴</h2></div><strong>{{ evidence.length }}</strong></div>
        <p v-if="currentVideo?.has_transcript" class="drawer-note">转录已保存，可直接继续提问。</p>
        <div v-if="evidence.length" class="timeline-list">
          <article v-for="item in evidence" :key="item.id" :class="['timeline-item', { 'is-cited': selectedEvidenceIds.has(item.id) }]">
            <div><span class="source-tag">{{ item.source || "ASR" }}</span><span>{{ formatSeconds(item.start_seconds) }}</span></div>
            <p>{{ item.text }}</p>
          </article>
        </div>
        <p v-else class="empty-state">当前视频还没有可用证据。</p>
      </aside>

      <section ref="reportsSection" class="clean-panel">
      <div class="clean-heading">
        <div><span class="section-label">VIDEO ANALYSIS</span><h2>报告与追问</h2></div>
        <span v-if="result" :class="['grounding-badge', result.grounded ? 'is-grounded' : 'is-refused']">{{ result.grounded ? "证据支持" : "证据不足" }}</span>
      </div>

      <form class="ask-form" @submit.prevent="ask">
        <textarea v-model="question" class="question-input" rows="3" placeholder="请输入你想从视频中确认的问题，例如：演讲者提出了哪些解决方案？"></textarea>
        <div class="clean-ask-footer"><span class="clean-hint">Agent 会先理解问题，再从 ASR/OCR 中筛选证据。</span><button class="primary-button" :disabled="loading || !question.trim()">{{ loading ? "处理中..." : "生成回答" }}</button></div>
      </form>

      <div v-if="analysisTask" class="task-progress-card report-progress-card">
        <div class="task-progress-top"><strong>正在生成视频分析报告</strong><span>{{ analysisProgress }}%</span></div>
        <div class="progress-track"><span :style="{ width: `${analysisProgress}%` }"></span></div>
        <div class="task-steps"><span class="active">理解问题</span><i></i><span class="active">检索证据</span><i></i><span class="active">生成报告</span></div>
      </div>

      <p v-if="loadingMessage" class="notice progress-notice">{{ loadingMessage }}</p>
      <p v-if="message" class="notice">{{ message }}</p>

      <article v-if="result?.error" class="clean-error"><div class="report-kicker">报告生成失败</div><p>{{ result.answer }}</p><span>修正模型配置后可重新提交问题。</span></article>
      <div v-if="reports.length" class="report-stream">
        <article v-for="item in reports" :key="item.id" :id="`report-${item.id}`" class="report-card-stream">
          <button class="stream-header" type="button" @click="toggleReport(item.id)"><div><span class="report-kicker">分析报告</span><h3>{{ item.question }}</h3><span class="stream-meta">{{ item.created_at }} · {{ (reportBody(item).evidence || reportBody(item).citations || []).length }} 条引用证据</span></div><span class="stream-header-side"><span :class="['grounding-badge', item.answerable ? 'is-grounded' : 'is-refused']">{{ item.answerable ? "证据支持" : "证据不足" }}</span><span class="stream-toggle">{{ expandedReportId === item.id ? "收起" : "展开" }}</span></span></button>
          <div v-if="expandedReportId === item.id" class="stream-body">
          <div class="report-toolbar"><span class="clean-hint">{{ item.report?.error || item.report?.report?.error ? "模型请求失败，已保留本次记录" : item.answerable ? "已通过证据校验" : "这条记录未找到足够证据" }}</span><button type="button" class="danger-button" :disabled="loading" @click="removeReport(item)">删除记录</button></div>
          <template v-if="reportBody(item).finalAnswer">
            <p class="report-answer">{{ reportBody(item).finalAnswer }}</p>
            <section v-if="reportBody(item).conclusions?.length" class="report-section"><h4>核心结论</h4><ul><li v-for="(conclusion, index) in reportBody(item).conclusions" :key="`${item.id}-conclusion-${index}`">{{ conclusion }}</li></ul></section>
            <section v-if="(reportBody(item).evidence || reportBody(item).citations)?.length" class="report-section"><h4>筛选证据</h4><div class="report-evidence-list"><article v-for="(citation, index) in (reportBody(item).evidence || reportBody(item).citations).slice(0, 9)" :key="`${item.id}-evidence-${index}`" class="report-evidence"><div class="report-evidence-meta"><strong>E{{ String(index + 1).padStart(2, '0') }}</strong><span class="source-tag">{{ citation.source || 'ASR' }}</span><span>{{ citation.timestamp || '未标注时间' }}</span></div><p>{{ citation.text || citation.content }}</p></article></div></section>
          </template>
          <div v-if="followUpLoading && followUpReportId === item.id" class="task-progress-card follow-up-progress-card" :id="`follow-up-${item.id}`"><div class="task-progress-top"><strong>{{ followUpProgressLabel }}</strong><span>{{ followUpProgress }}%</span></div><div class="progress-track"><span :style="{ width: `${followUpProgress}%` }"></span></div><div class="task-steps"><span class="active">检索相关证据</span><i></i><span class="active">生成段落回答</span></div></div>
          <div v-for="followUp in item.follow_ups || []" :key="followUp.id" :id="`follow-up-record-${followUp.id}`" class="follow-up-card"><div class="follow-up-heading"><span class="report-kicker">追问</span><button type="button" class="text-danger-button" :disabled="loading" @click="removeFollowUp(item, followUp)">删除</button></div><strong>{{ followUp.question }}</strong><p>{{ followUp.answer }}</p></div>
          <form class="inline-follow-up" @submit.prevent="submitFollowUp(item)"><input v-model="followUpDrafts[item.id]" placeholder="针对这份报告继续追问..." :disabled="loading" /><button class="secondary-button" :disabled="loading || !followUpDrafts[item.id]?.trim()">追问</button></form>
          </div>
        </article>
      </div>
      <div v-else-if="!result?.error" class="result-placeholder"><strong>还没有报告</strong><span>先提交一个问题，系统会筛选证据并生成独立报告。</span></div>
      </section>
    </section>
  </main>
</template>
