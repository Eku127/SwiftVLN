import { useConsoleController } from "../hooks/useConsoleController";
import { ActionPanel } from "../components/ActionPanel";
import { FlightControlPanel } from "../components/FlightControlPanel";
import { Header } from "../components/Header";
import { ImagePanel } from "../components/ImagePanel";
import { InstructionPanel } from "../components/InstructionPanel";
import { LogPanel } from "../components/LogPanel";

export function ConsolePage() {
  const controller = useConsoleController();

  return (
    <div className="console-layout">
      <Header
        className="console-layout__header"
        apiHealthy={controller.apiHealthy}
        modelReady={controller.modelReady}
        rtmpActive={controller.rtmpActive}
        flightBackendHealthy={controller.flightBackendHealthy}
        sessionId={controller.sessionId}
        phaseLabel={controller.phaseLabel}
        settingsBusy={controller.settingsBusy}
        onRefreshRtmp={controller.onRefreshRtmp}
      />

      <InstructionPanel
        className="console-layout__instruction"
        instruction={controller.instruction}
        onInstructionChange={controller.setInstruction}
        maxLength={1000}
        inferenceBusy={controller.inferenceBusy}
        inferenceEnabled={controller.inferenceEnabled}
        onInference={controller.onInference}
      />

      <FlightControlPanel
        className="console-layout__flight"
        busy={controller.flightBusy}
        loggedIn={controller.loggedIn}
        drcReady={controller.flightReady}
        aircraftPose={controller.aircraftPose}
        defaultRcSn={controller.defaultRcSn}
        defaultDeviceSn={controller.defaultDeviceSn}
        defaultLoginUsername={controller.defaultLoginUsername}
        defaultLoginPassword={controller.defaultLoginPassword}
        defaultLoginFlag={controller.defaultLoginFlag}
        onRegisterDevice={controller.onRegisterDevice}
        onLogin={controller.onLoginFlightSystem}
        onAcquireControl={controller.onAcquireFlightControl}
        osdRefreshBusy={controller.osdRefreshBusy}
        onRefreshOsd={controller.onRefreshOsd}
      />

      <ImagePanel
        className="console-layout__raw"
        variant="raw"
        title="RTMP 原始抽帧"
        subtitle={controller.rawMeta}
        imageUrl={controller.rawImageUrl}
        placeholder="等待 RTMP 帧..."
      />

      <ImagePanel
        className="console-layout__model"
        variant="model"
        title="模型输入 · 448 × 448"
        subtitle={controller.modelInputMeta}
        imageUrl={controller.modelInputUrl}
        badge="RGB"
        placeholder="请先执行 inference"
      />

      <ActionPanel
        className="console-layout__action"
        flightBusy={controller.flightBusy}
        executeEnabled={controller.executeEnabled}
        skipEnabled={controller.skipEnabled}
        flightProgress={controller.flightProgress}
        latestStickTask={controller.latestStickTask}
        inferenceNextAction={controller.inferenceNextAction}
        inferenceRemainingActions={controller.inferenceRemainingActions}
        inferencePerformed={controller.inferencePerformed}
        queueSlots={controller.actionQueue.slots}
        queueLength={controller.actionQueue.queueLength}
        currentIndex={controller.actionQueue.currentIndex}
        stopEngaged={controller.stopEngaged}
        stopModalOpen={controller.stopModalOpen}
        onDismissStopModal={controller.dismissStopModal}
        currentStickTaskId={controller.currentStickTaskId}
        stickRefreshBusy={controller.stickRefreshBusy}
        onRunOneStep={controller.onRunOneStep}
        onSkipCurrentStep={controller.onSkipCurrentStep}
        onRefreshStickTask={controller.onRefreshStickTask}
        onEmergencyStop={controller.onEmergencyStop}
        autoFlightEnabled={controller.autoFlightEnabled}
        onAutoFlightChange={controller.onAutoFlightChange}
        autoFlightCountdownSec={controller.autoFlightCountdownSec}
        autoFlightPrereqModalOpen={controller.autoFlightPrereqModalOpen}
        onDismissAutoFlightPrereqModal={controller.dismissAutoFlightPrereqModal}
        autoFlightFailureModalOpen={controller.autoFlightFailureModalOpen}
        onAutoFlightFailureRetry={controller.onAutoFlightFailureRetry}
        onAutoFlightFailureSkip={controller.onAutoFlightFailureSkip}
        autoFlightInferenceFailModalOpen={controller.autoFlightInferenceFailModalOpen}
        onDismissAutoFlightInferenceFailModal={controller.dismissAutoFlightInferenceFailModal}
      />

      <LogPanel
        className="console-layout__log"
        lines={controller.logLines}
        autoScroll={controller.autoScrollLogs}
        onAutoScrollChange={controller.setAutoScrollLogs}
        onClear={controller.clearLogs}
      />
    </div>
  );
}
