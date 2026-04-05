# -*- coding: utf-8 -*-
"""Workflow module"""
from trinity.common.workflows.agentscope.react.react_workflow import (
    AgentScopeReActWorkflow,
)
from trinity.common.workflows.agentscope_workflow import AgentScopeWorkflowAdapter
from trinity.common.workflows.customized_math_workflows import (
    AsyncMathBoxedWorkflow,
    MathBoxedWorkflow,
)
from trinity.common.workflows.customized_toolcall_workflows import ToolCallWorkflow
from trinity.common.workflows.envs.agentscope.agentscopev0_react_workflow import (  # will be deprecated soon
    AgentScopeV0ReactMathWorkflow,
)
from trinity.common.workflows.envs.agentscope.agentscopev1_react_workflow import (
    AgentScopeReactMathWorkflow,
)
from trinity.common.workflows.envs.agentscope.agentscopev1_search_workflow import (
    AgentScopeV1ReactSearchWorkflow,
)
from trinity.common.workflows.envs.alfworld.alfworld_workflow import (
    AlfworldWorkflow,
    StepWiseAlfworldWorkflow,
)
from trinity.common.workflows.envs.alfworld.RAFT_alfworld_workflow import (
    RAFTAlfworldWorkflow,
)
from trinity.common.workflows.envs.alfworld.RAFT_reflect_alfworld_workflow import (
    RAFTReflectAlfworldWorkflow,
)
from trinity.common.workflows.envs.email_searcher.workflow import EmailSearchWorkflow
from trinity.common.workflows.envs.R3L.alfworld.critique_grpo_workflow import (
    CritiqueGRPOAlfworldWorkflow,
)

# Alfworld R3L workflows
from trinity.common.workflows.envs.R3L.alfworld.dapo_workflow import (
    DAPOAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.grpo_workflow import (
    GRPOBaselineAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.opmd_workflow import (
    OPMDBaselineAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.R3L_w_o_credit_workflow import (
    R3LAlfworldWoCreditWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.R3L_workflow import R3LAlfworldWorkflow
from trinity.common.workflows.envs.R3L.alfworld.raft_workflow import (
    RAFTBaselineAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.reflect_grpo_workflow import (
    ReflectGRPOAlfworldWorkflow,
)

# Step-level Alfworld R3L workflows (Qwen3)
from trinity.common.workflows.envs.R3L.alfworld.step_critique_grpo_workflow import (
    StepLevelCritiqueGRPOAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.step_grpo_workflow import (
    StepLevelAlfworldBaseWorkflow,
    StepLevelDAPOAlfworldWorkflow,
    StepLevelGRPOAlfworldWorkflow,
    StepLevelGSPOAlfworldWorkflow,
    StepLevelOPMDAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.step_R3L_workflow import (
    StepLevelR3LAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.step_raft_workflow import (
    StepLevelRAFTAlfworldWorkflow,
)
from trinity.common.workflows.envs.R3L.alfworld.step_reflect_grpo_workflow import (
    StepLevelReflectGRPOAlfworldWorkflow,
)

# Step-level WebShop R3L workflows (Qwen3)
from trinity.common.workflows.envs.R3L.webshop.step_critique_grpo_workflow import (
    StepLevelCritiqueGRPOWebshopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.step_grpo_workflow import (
    StepLevelDAPOWebshopWorkflow,
    StepLevelGRPOWebshopWorkflow,
    StepLevelGSPOWebshopWorkflow,
    StepLevelOPMDWebshopWorkflow,
    StepLevelWebshopBaseWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.step_R3L_workflow import (
    StepLevelR3LWebshopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.step_raft_workflow import (
    StepLevelRAFTWebshopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.step_reflect_grpo_workflow import (
    StepLevelReflectGRPOWebshopWorkflow,
)

# Step-level ScienceWorld R3L workflows (Qwen3)
from trinity.common.workflows.envs.R3L.scienceworld.step_critique_grpo_workflow import (
    StepLevelCritiqueGRPOScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.step_grpo_workflow import (
    StepLevelDAPOScienceWorldWorkflow,
    StepLevelGRPOScienceWorldWorkflow,
    StepLevelGSPOScienceWorldWorkflow,
    StepLevelOPMDScienceWorldWorkflow,
    StepLevelScienceWorldBaseWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.step_R3L_workflow import (
    StepLevelR3LScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.step_raft_workflow import (
    StepLevelRAFTScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.step_reflect_grpo_workflow import (
    StepLevelReflectGRPOScienceWorldWorkflow,
)

# Step-level DAPO R3L workflows (Qwen3)
from trinity.common.workflows.envs.R3L.dapo.step_critique_grpo_workflow import (
    StepLevelCritiqueGRPODapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.step_grpo_workflow import (
    StepLevelDAPODapoWorkflow,
    StepLevelDapoBaseWorkflow,
    StepLevelGRPODapoWorkflow,
    StepLevelGSPODapoWorkflow,
    StepLevelOPMDDapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.step_R3L_workflow import (
    StepLevelR3LDapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.step_raft_workflow import (
    StepLevelRAFTDapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.step_reflect_grpo_workflow import (
    StepLevelReflectGRPODapoWorkflow,
)

# Countdown R3L workflows
from trinity.common.workflows.envs.R3L.countdown.dapo_workflow import (
    DAPOCountdownWorkflow,
)
from trinity.common.workflows.envs.R3L.countdown.grpo_workflow import (
    GRPOBaselineCountdownWorkflow,
)
from trinity.common.workflows.envs.R3L.countdown.opmd_workflow import (
    OPMDBaselineCountdownWorkflow,
)
from trinity.common.workflows.envs.R3L.countdown.R3L_workflow import (
    R3LCountdownWorkflow,
)
from trinity.common.workflows.envs.R3L.countdown.raft_workflow import (
    RAFTBaselineCountdownWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.critique_grpo_workflow import (
    CritiqueGRPODapoWorkflow,
)

# DAPO R3L workflows
from trinity.common.workflows.envs.R3L.dapo.dapo_workflow import DAPODapoWorkflow
from trinity.common.workflows.envs.R3L.dapo.grpo_workflow import (
    GRPOBaselineDapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.opmd_workflow import (
    OPMDBaselineDapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.R3L_w_o_credit_workflow import (
    R3LDapoWoCreditWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.R3L_workflow import R3LDapoWorkflow
from trinity.common.workflows.envs.R3L.dapo.raft_workflow import (
    RAFTBaselineDapoWorkflow,
)
from trinity.common.workflows.envs.R3L.dapo.reflect_grpo_workflow import (
    ReflectGRPODapoWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.critique_grpo_workflow import (
    CritiqueGRPOScienceWorldWorkflow,
)

# ScienceWorld R3L workflows
from trinity.common.workflows.envs.R3L.scienceworld.dapo_workflow import (
    DAPOScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.grpo_workflow import (
    GRPOBaselineScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.opmd_workflow import (
    OPMDBaselineScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.R3L_w_o_credit_workflow import (
    R3LScienceWorldWoCreditWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.R3L_workflow import (
    R3LScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.raft_workflow import (
    RAFTBaselineScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.scienceworld.reflect_grpo_workflow import (
    ReflectGRPOScienceWorldWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.critique_grpo_workflow import (
    CritiqueGRPOWebShopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.dapo_workflow import DAPOWebshopWorkflow
from trinity.common.workflows.envs.R3L.webshop.grpo_workflow import (
    GRPOBaselineWebshopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.opmd_workflow import (
    OPMDBaselineWebshopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.R3L_w_o_credit_workflow import (
    R3LWebshopWoCreditWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.R3L_workflow import R3LWebshopWorkflow
from trinity.common.workflows.envs.R3L.webshop.raft_workflow import (
    RAFTBaselineWebshopWorkflow,
)
from trinity.common.workflows.envs.R3L.webshop.reflect_grpo_workflow import (
    ReflectGRPOWebshopWorkflow,
)
# Qwen Math workflows (boxed format)
from trinity.common.workflows.envs.R3L.qwen_math.critique_grpo_workflow import (
    CritiqueGRPOQwenMathWorkflow,
)
from trinity.common.workflows.envs.R3L.qwen_math.dapo_workflow import DAPOQwenMathWorkflow
from trinity.common.workflows.envs.R3L.qwen_math.grpo_workflow import (
    GRPOBaselineQwenMathWorkflow,
)
from trinity.common.workflows.envs.R3L.qwen_math.opmd_workflow import (
    OPMDBaselineQwenMathWorkflow,
)
from trinity.common.workflows.envs.R3L.qwen_math.R3L_w_o_credit_workflow import (
    R3LQwenMathWoCreditWorkflow,
)
from trinity.common.workflows.envs.R3L.qwen_math.R3L_workflow import R3LQwenMathWorkflow
from trinity.common.workflows.envs.R3L.qwen_math.raft_workflow import (
    RAFTBaselineQwenMathWorkflow,
)
from trinity.common.workflows.envs.R3L.qwen_math.reflect_grpo_workflow import (
    ReflectGRPOQwenMathWorkflow,
)
from trinity.common.workflows.envs.sciworld.sciworld_workflow import SciWorldWorkflow
from trinity.common.workflows.envs.webshop.webshop_workflow import WebShopWorkflow

#
from trinity.common.workflows.eval_workflow import (
    AsyncMathEvalWorkflow,
    MathEvalWorkflow,
)
from trinity.common.workflows.math_rm_workflow import (
    AsyncMathRMWorkflow,
    MathRMWorkflow,
)
from trinity.common.workflows.math_ruler_workflow import (
    AsyncMathRULERWorkflow,
    MathRULERWorkflow,
)
from trinity.common.workflows.math_trainable_ruler_workflow import (
    MathTrainableRULERWorkflow,
)
from trinity.common.workflows.rubric_judge_workflow import RubricJudgeWorkflow
from trinity.common.workflows.simple_mm_workflow import (
    AsyncSimpleMMWorkflow,
    SimpleMMWorkflow,
)
from trinity.common.workflows.workflow import (
    WORKFLOWS,
    AsyncMathWorkflow,
    AsyncSimpleWorkflow,
    MathWorkflow,
    SimpleWorkflow,
    Task,
    Workflow,
)

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
    "AsyncSimpleWorkflow",
    "SimpleWorkflow",
    "AsyncMathWorkflow",
    "MathWorkflow",
    "WebShopWorkflow",
    "R3LWebshopWorkflow",
    "R3LWebshopWoCreditWorkflow",
    "GRPOBaselineWebshopWorkflow",
    "OPMDBaselineWebshopWorkflow",
    "RAFTBaselineWebshopWorkflow",
    "DAPOWebshopWorkflow",
    "ReflectGRPOWebshopWorkflow",
    "CritiqueGRPOWebShopWorkflow",
    # Alfworld R3L workflows
    "R3LAlfworldWorkflow",
    "R3LAlfworldWoCreditWorkflow",
    "GRPOBaselineAlfworldWorkflow",
    "OPMDBaselineAlfworldWorkflow",
    "RAFTBaselineAlfworldWorkflow",
    "DAPOAlfworldWorkflow",
    "ReflectGRPOAlfworldWorkflow",
    "CritiqueGRPOAlfworldWorkflow",
    # Step-level Alfworld R3L workflows (Qwen3)
    "StepLevelAlfworldBaseWorkflow",
    "StepLevelGRPOAlfworldWorkflow",
    "StepLevelOPMDAlfworldWorkflow",
    "StepLevelGSPOAlfworldWorkflow",
    "StepLevelDAPOAlfworldWorkflow",
    "StepLevelRAFTAlfworldWorkflow",
    "StepLevelReflectGRPOAlfworldWorkflow",
    "StepLevelCritiqueGRPOAlfworldWorkflow",
    "StepLevelR3LAlfworldWorkflow",
    # Step-level WebShop R3L workflows (Qwen3)
    "StepLevelWebshopBaseWorkflow",
    "StepLevelGRPOWebshopWorkflow",
    "StepLevelOPMDWebshopWorkflow",
    "StepLevelGSPOWebshopWorkflow",
    "StepLevelDAPOWebshopWorkflow",
    "StepLevelRAFTWebshopWorkflow",
    "StepLevelReflectGRPOWebshopWorkflow",
    "StepLevelCritiqueGRPOWebshopWorkflow",
    "StepLevelR3LWebshopWorkflow",
    # Step-level ScienceWorld R3L workflows (Qwen3)
    "StepLevelScienceWorldBaseWorkflow",
    "StepLevelGRPOScienceWorldWorkflow",
    "StepLevelOPMDScienceWorldWorkflow",
    "StepLevelGSPOScienceWorldWorkflow",
    "StepLevelDAPOScienceWorldWorkflow",
    "StepLevelRAFTScienceWorldWorkflow",
    "StepLevelReflectGRPOScienceWorldWorkflow",
    "StepLevelCritiqueGRPOScienceWorldWorkflow",
    "StepLevelR3LScienceWorldWorkflow",
    # Step-level DAPO R3L workflows (Qwen3)
    "StepLevelDapoBaseWorkflow",
    "StepLevelGRPODapoWorkflow",
    "StepLevelOPMDDapoWorkflow",
    "StepLevelGSPODapoWorkflow",
    "StepLevelDAPODapoWorkflow",
    "StepLevelRAFTDapoWorkflow",
    "StepLevelReflectGRPODapoWorkflow",
    "StepLevelCritiqueGRPODapoWorkflow",
    "StepLevelR3LDapoWorkflow",
    # DAPO R3L workflows
    "R3LDapoWorkflow",
    "R3LDapoWoCreditWorkflow",
    "GRPOBaselineDapoWorkflow",
    "OPMDBaselineDapoWorkflow",
    "RAFTBaselineDapoWorkflow",
    "DAPODapoWorkflow",
    "ReflectGRPODapoWorkflow",
    "CritiqueGRPODapoWorkflow",
    # ScienceWorld R3L workflows
    "R3LScienceWorldWorkflow",
    "R3LScienceWorldWoCreditWorkflow",
    "GRPOBaselineScienceWorldWorkflow",
    "OPMDBaselineScienceWorldWorkflow",
    "RAFTBaselineScienceWorldWorkflow",
    "DAPOScienceWorldWorkflow",
    "ReflectGRPOScienceWorldWorkflow",
    "CritiqueGRPOScienceWorldWorkflow",
    # Countdown R3L workflows
    "R3LCountdownWorkflow",
    "GRPOBaselineCountdownWorkflow",
    "OPMDBaselineCountdownWorkflow",
    "RAFTBaselineCountdownWorkflow",
    "DAPOCountdownWorkflow",
    # Original workflows
    "AlfworldWorkflow",
    "StepWiseAlfworldWorkflow",
    "RAFTAlfworldWorkflow",
    "RAFTReflectAlfworldWorkflow",
    "SciWorldWorkflow",
    "AsyncMathBoxedWorkflow",
    "MathBoxedWorkflow",
    "AsyncMathRMWorkflow",
    "MathRMWorkflow",
    "ToolCallWorkflow",
    "AsyncMathEvalWorkflow",
    "MathEvalWorkflow",
    "AgentScopeV0ReactMathWorkflow",  # will be deprecated soon
    "AgentScopeReactMathWorkflow",
    "AgentScopeV1ReactSearchWorkflow",
    "AgentScopeReActWorkflow",
    "EmailSearchWorkflow",
    "AsyncMathRULERWorkflow",
    "MathRULERWorkflow",
    "MathTrainableRULERWorkflow",
    "AsyncSimpleMMWorkflow",
    "SimpleMMWorkflow",
    "RubricJudgeWorkflow",
    "AgentScopeWorkflowAdapter",
    # Qwen Math workflows (boxed format)
    "GRPOBaselineQwenMathWorkflow",
    "DAPOQwenMathWorkflow",
    "RAFTBaselineQwenMathWorkflow",
    "OPMDBaselineQwenMathWorkflow",
    "CritiqueGRPOQwenMathWorkflow",
    "ReflectGRPOQwenMathWorkflow",
    "R3LQwenMathWorkflow",
    "R3LQwenMathWoCreditWorkflow",
]
