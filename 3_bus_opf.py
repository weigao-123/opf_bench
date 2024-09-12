"""
This code example is to demonstrate the bus injection model and branch flow model for optimal power flow.
Only for educational purposes, the code is not optimized for performance.

note: the original ac opf is a nonlinear problem, and the solver will run into local minima
even with the same model but with different formulations, the results will be different
this could be proven by restrict the bus voltage to 1.0, and the results will be the same.
"""

import pandas as pd
import pyomo.environ as pyo


def bus_injection_model(
    bus, gen_buses, y_bus_df, slack_bus, load_p, load_q, time, bus_v_bounds=None
):
    slack = list(slack_bus.keys())[0]
    # Model
    if bus_v_bounds is None:
        bus_v_bounds = {}
    model = pyo.ConcreteModel()

    # Sets
    model.bus = pyo.Set(initialize=bus)
    model.time = pyo.Set(initialize=time)

    # Parameters
    # network demand
    model.load_p = pyo.Param(
        model.bus,
        model.time,
        within=pyo.NonNegativeReals,
        initialize=lambda model, i, t: load_p.get(i, 0),
    )
    model.load_q = pyo.Param(
        model.bus,
        model.time,
        within=pyo.NonNegativeReals,
        initialize=lambda model, i, t: load_q.get(i, 0),
    )

    # network parameters
    model.bus_ij_g = pyo.Param(
        model.bus,
        model.bus,
        within=pyo.Reals,
        initialize=lambda model, i, j: (
            y_bus_df.loc[i, j].real
            if (i in y_bus_df.index and j in y_bus_df.columns)
            else 0.0
        ),
    )
    model.bus_ij_b = pyo.Param(
        model.bus,
        model.bus,
        within=pyo.Reals,
        initialize=lambda model, i, j: (
            y_bus_df.loc[i, j].imag
            if (i in y_bus_df.index and j in y_bus_df.columns)
            else 0.0
        ),
    )

    # network operational parameters
    model.bus_v_bounds = pyo.Param(
        model.bus,
        within=pyo.Any,
        initialize=lambda model, i: bus_v_bounds.get(i, (0.9, 1.1)),
    )

    # gen operational parameters
    model.gen_p_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (gen_buses.get(i, {}).get("p_min", 0), gen_buses.get(i, {}).get("p_max", 0))
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )
    model.gen_q_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (
                gen_buses.get(i, {}).get("q_min", 0),
                gen_buses.get(i, {}).get("q_max", 0),
            )
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    ## Variables
    model.gen_p = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_p_bounds[i, t],
    )

    model.gen_q = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_q_bounds[i, t],
    )

    # network operational variables
    model.bus_v = pyo.Var(
        model.bus,
        model.time,
        initialize=1,
        within=pyo.NonNegativeReals,
        bounds=lambda model, i, t: model.bus_v_bounds[i],
    )
    model.bus_theta = pyo.Var(
        model.bus, model.time, within=pyo.Reals, initialize=0, bounds=(-3.1416, 3.1416)
    )
    model.bus_p = pyo.Var(model.bus, model.time, within=pyo.Reals)
    model.bus_q = pyo.Var(model.bus, model.time, within=pyo.Reals)

    # Set slack bus voltage and angle
    slack = list(slack_bus.keys())[0]

    # for t in model.time:
    #     model.bus_v[slack, t].fix(slack_bus[slack]["v"])
    #     model.bus_theta[slack, t].fix(slack_bus[slack]["theta"])
    #
    # for t in model.time:
    #     for gen in gen_buses:
    #         model.bus_v[gen, t].fix(gen_buses[gen]["v"])

    # Constraints
    # power flow equations
    # @model.Constraint(model.bus, model.time)
    def bus_active_power_flow_rule(model, i, t):
        return model.bus_p[i, t] == sum(
            model.bus_v[i, t]
            * model.bus_v[j, t]
            * (
                model.bus_ij_g[i, j]
                * pyo.cos(model.bus_theta[i, t] - model.bus_theta[j, t])
                + model.bus_ij_b[i, j]
                * pyo.sin(model.bus_theta[i, t] - model.bus_theta[j, t])
            )
            for j in model.bus
        )

    model.cons_bus_active_power_flow_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_active_power_flow_rule
    )

    # @model.Constraint(model.bus, model.time)
    def bus_reactive_power_flow_rule(model, i, t):
        return model.bus_q[i, t] == sum(
            model.bus_v[i, t]
            * model.bus_v[j, t]
            * (
                model.bus_ij_g[i, j]
                * pyo.sin(model.bus_theta[i, t] - model.bus_theta[j, t])
                - model.bus_ij_b[i, j]
                * pyo.cos(model.bus_theta[i, t] - model.bus_theta[j, t])
            )
            for j in model.bus
        )

    model.cons_bus_reactive_power_flow_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_reactive_power_flow_rule
    )

    # @model.Constraint(model.bus, model.time)
    def bus_active_power_balance_rule(model, i, t):
        return model.bus_p[i, t] == model.gen_p[i, t] - model.load_p[i, t]

    model.cons_bus_active_power_balance_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_active_power_balance_rule
    )

    # @model.Constraint(model.bus, model.time)
    def bus_reactive_power_balance_rule(model, i, t):
        return model.bus_q[i, t] == model.gen_q[i, t] - model.load_q[i, t]

    model.cons_bus_reactive_power_balance_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_reactive_power_balance_rule
    )

    # Objective
    # @model.Objective() minimum power loss
    def obj_rule(model):
        return sum((model.bus_v[i, t] - 1) ** 2 for i in model.bus for t in model.time)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


def branch_flow_model_line_current(
    bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds=None
):
    slack = list(slack_bus.keys())[0]
    if bus_v_bounds is None:
        bus_v_bounds = {}
    model = pyo.ConcreteModel()

    # Sets
    model.bus = pyo.Set(initialize=bus)
    model.time = pyo.Set(initialize=time)
    model.line = pyo.Set(initialize=line_data.keys(), dimen=2)

    # Parameters
    model.load_p = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_p.get(i, 0),
    )
    model.load_q = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_q.get(i, 0),
    )

    model.line_R = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["R"]
    )
    model.line_X = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["X"]
    )

    model.gen_p_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (gen_buses.get(i, {}).get("p_min", 0), gen_buses.get(i, {}).get("p_max", 0))
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.gen_q_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (
                gen_buses.get(i, {}).get("q_min", 0),
                gen_buses.get(i, {}).get("q_max", 0),
            )
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.bus_v_bounds = pyo.Param(
        model.bus,
        within=pyo.Any,
        initialize=lambda model, i: bus_v_bounds.get(i, (0.9, 1.1)),
    )

    model.bus_theta_bounds = pyo.Param(
        model.bus, within=pyo.Any, initialize=lambda model, i: (-3.1416, 3.1416)
    )

    # Variables
    model.gen_p = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_p_bounds[i, t],
    )

    model.gen_q = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_q_bounds[i, t],
    )

    # Network operational variables
    model.bus_v = pyo.Var(
        model.bus,
        model.time,
        within=pyo.NonNegativeReals,
        initialize=1,
        bounds=lambda model, i, t: model.bus_v_bounds[i],
    )
    model.bus_theta = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=0,
        bounds=lambda model, i, t: model.bus_theta_bounds[i],
    )
    model.Iij = pyo.Var(model.line, model.time, within=pyo.Reals)  # Line current
    model.Iij_theta = pyo.Var(
        model.line, model.time, within=pyo.Reals
    )  # Line current angle

    # Set slack bus voltage and angle
    # slack = list(slack_bus.keys())[0]

    # for t in model.time:
    #     model.bus_v[slack, t].fix(slack_bus[slack]["v"])
    #     model.bus_theta[slack, t].fix(slack_bus[slack]["theta"])
    #
    # for t in model.time:
    #     for gen in gen_buses:
    #         model.bus_v[gen, t].fix(gen_buses[gen]["v"])

    # Constraints  branch flow model: (https://arxiv.org/pdf/1204.4865, Branch Flow Model: Relaxations and
    # Convexification) !!! Notice: we do not want to model the line current in the optimization problem,
    # because internally, the line current could be calculated from bus voltage and line flow, !!! and if we model
    # the line current, ideally, we could still solve this problem without any difference, but since this is a
    # non-linear problem, the solver may run into numerical issues !!! and cause the problem to be infeasible.
    # Therefore, we will model the line flow directly, and the line current will be calculated internally by the
    # solver. Ohm's law
    # @model.Constraint(model.line, model.time)
    def ohms_law_rule_mag(model, i, j, t):
        Rij = model.line_R[i, j]
        Xij = model.line_X[i, j]
        Vi = model.bus_v[i, t]
        Vj = model.bus_v[j, t]
        V_thetai = model.bus_theta[i, t]
        V_thetaj = model.bus_theta[j, t]

        """
        # vi = Vi * pyo.cos(V_thetai) + Vi * pyo.sin(V_thetai) * 1j
        # vj = Vj * pyo.cos(V_thetaj) + Vj * pyo.sin(V_thetaj) * 1j
        # Iij = (vi - vj) / (Rij + Xij * 1j)
        # Iij_conj = pyo.conjugate(Iij)

        the above equations are equivalent to the following:
        """
        Vi_re = Vi * pyo.cos(V_thetai)
        Vi_im = Vi * pyo.sin(V_thetai)
        Vj_re = Vj * pyo.cos(V_thetaj)
        Vj_im = Vj * pyo.sin(V_thetaj)
        Vi_Vj_re = Vi_re - Vj_re
        Vi_Vj_im = Vi_im - Vj_im
        Iij_re = (Vi_Vj_re * Rij + Vi_Vj_im * Xij) / (Rij**2 + Xij**2)
        # Iij_im = (Vi_Vj_im * Rij - Vi_Vj_re * Xij) / (Rij**2 + Xij**2)

        Iij = model.Iij[i, j, t]
        Iij_theta = model.Iij_theta[i, j, t]
        return Iij * pyo.cos(Iij_theta) == Iij_re

    model.cons_ohms_law_rule_mag = pyo.Constraint(
        model.line, model.time, rule=ohms_law_rule_mag
    )

    # Power balance constraints
    # @model.Constraint(model.bus, model.time)
    def bus_active_power_balance_rule(model, i, t):
        incoming = 0
        outgoing = 0
        for k, l in model.line:
            Rkl = model.line_R[k, l]
            Vk = model.bus_v[l, t]
            V_thetak = model.bus_theta[k, t]

            Vk_re = Vk * pyo.cos(V_thetak)
            Vk_im = Vk * pyo.sin(V_thetak)

            Ikl_re = model.Iij[k, l, t] * pyo.cos(model.Iij_theta[k, l, t])
            Ikl_im = model.Iij[k, l, t] * pyo.sin(model.Iij_theta[k, l, t])
            Pkl = Vk_re * Ikl_re + Vk_im * Ikl_im
            if l == i:
                incoming = incoming + Pkl - model.Iij[k, l, t] ** 2 * Rkl
            if k == i:
                outgoing = outgoing + Pkl
        return (
            model.gen_p[i, t] - model.load_p[i, t] == outgoing - incoming
        )  # incoming(after line_losses) - outgoing == gen_p - load_p

    model.cons_bus_active_power_balance_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_active_power_balance_rule
    )

    # @model.Constraint(model.bus, model.time)
    def bus_reactive_power_balance_rule(model, i, t):
        incoming = 0
        outgoing = 0
        for k, l in model.line:
            Xkl = model.line_X[k, l]
            Vk = model.bus_v[l, t]
            V_thetak = model.bus_theta[k, t]

            Vk_re = Vk * pyo.cos(V_thetak)
            Vk_im = Vk * pyo.sin(V_thetak)

            Ikl_re = model.Iij[k, l, t] * pyo.cos(model.Iij_theta[k, l, t])
            Ikl_im = model.Iij[k, l, t] * pyo.sin(model.Iij_theta[k, l, t])
            Qkl = Vk_im * Ikl_re - Vk_re * Ikl_im
            if l == i:
                incoming = incoming + Qkl - model.Iij[k, l, t] ** 2 * Xkl
            if k == i:
                outgoing = outgoing + Qkl
        return (
            model.gen_q[i, t] - model.load_q[i, t] == outgoing - incoming
        )  # incoming(after line_losses) - outgoing == gen_q - load_q

    model.cons_bus_reactive_power_balance_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_reactive_power_balance_rule
    )

    # Objective
    # @model.Objective(sense=pyo.minimize)
    def obj_rule(model):
        return sum((model.bus_v[i, t] - 1) ** 2 for i in model.bus for t in model.time)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


def branch_flow_model_line_flow(
    bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds=None
):
    slack = list(slack_bus.keys())[0]
    if bus_v_bounds is None:
        bus_v_bounds = {}
    model = pyo.ConcreteModel()

    # Sets
    model.bus = pyo.Set(initialize=bus)
    model.time = pyo.Set(initialize=time)
    model.line = pyo.Set(initialize=line_data.keys(), dimen=2)

    # Parameters
    model.load_p = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_p.get(i, 0),
    )
    model.load_q = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_q.get(i, 0),
    )

    model.line_R = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["R"]
    )
    model.line_X = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["X"]
    )

    model.gen_p_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (gen_buses.get(i, {}).get("p_min", 0), gen_buses.get(i, {}).get("p_max", 0))
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.gen_q_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (
                gen_buses.get(i, {}).get("q_min", 0),
                gen_buses.get(i, {}).get("q_max", 0),
            )
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.bus_v_bounds = pyo.Param(
        model.bus,
        within=pyo.Any,
        initialize=lambda model, i: bus_v_bounds.get(i, (0.9, 1.1)),
    )

    model.bus_theta_bounds = pyo.Param(
        model.bus, within=pyo.Any, initialize=lambda model, i: (-3.1416, 3.1416)
    )

    # Variables
    model.gen_p = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_p_bounds[i, t],
    )

    model.gen_q = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_q_bounds[i, t],
    )

    # Network operational variables
    model.bus_v = pyo.Var(
        model.bus,
        model.time,
        within=pyo.NonNegativeReals,
        initialize=1,
        bounds=lambda model, i, t: model.bus_v_bounds[i],
    )
    model.bus_theta = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=0,
        bounds=lambda model, i, t: model.bus_theta_bounds[i],
    )
    # model.Iij = pyo.Var(model.line, model.time, within=pyo.Reals)  # Line current
    # model.Iij_theta = pyo.Var(model.line, model.time, within=pyo.Reals)  # Line current angle
    model.Pij = pyo.Var(model.line, model.time, within=pyo.Reals)  # Real power flow
    model.Qij = pyo.Var(model.line, model.time, within=pyo.Reals)  # Reactive power flow

    # Set slack bus voltage and angle
    # slack = list(slack_bus.keys())[0]

    # for t in model.time:
    #     model.bus_v[slack, t].fix(slack_bus[slack]["v"])
    #     model.bus_theta[slack, t].fix(slack_bus[slack]["theta"])
    #
    # for t in model.time:
    #     for gen in gen_buses:
    #         model.bus_v[gen, t].fix(gen_buses[gen]["v"])

    # Constraints branch flow model: (https://arxiv.org/pdf/1204.4865, Branch Flow Model: Relaxations and
    # Convexification) !!! Notice: we do not want to model the line current in the optimization problem,
    # because internally, the line current could be calculated from bus voltage and line flow, !!! and if we model
    # the line current, ideally, we could still solve this problem without any difference, but since this is a
    # non-linear problem, the solver may run into numerical issues !!! and cause the problem to be infeasible.
    # Therefore, we will model the line flow directly, and the line current will be calculated internally by the
    # solver.

    # Branch flow equations for real and reactive power
    # @model.Constraint(model.line, model.time)
    def branch_flow_real_rule(model, i, j, t):
        Rij = model.line_R[i, j]
        Xij = model.line_X[i, j]
        Vi = model.bus_v[i, t]
        Vj = model.bus_v[j, t]
        V_thetai = model.bus_theta[i, t]
        V_thetaj = model.bus_theta[j, t]

        """
        # vi = Vi * pyo.cos(V_thetai) + Vi * pyo.sin(V_thetai) * 1j
        # vj = Vj * pyo.cos(V_thetaj) + Vj * pyo.sin(V_thetaj) * 1j
        # Iij = (vi - vj) / (Rij + Xij * 1j)
        # Iij_conj = pyo.conjugate(Iij)

        the above equations are equivalent to the following:
        """
        Vi_re = Vi * pyo.cos(V_thetai)
        Vi_im = Vi * pyo.sin(V_thetai)
        Vj_re = Vj * pyo.cos(V_thetaj)
        Vj_im = Vj * pyo.sin(V_thetaj)
        Vi_Vj_re = Vi_re - Vj_re
        Vi_Vj_im = Vi_im - Vj_im
        Iij_re = (Vi_Vj_re * Rij + Vi_Vj_im * Xij) / (Rij**2 + Xij**2)
        Iij_im = (Vi_Vj_im * Rij - Vi_Vj_re * Xij) / (Rij**2 + Xij**2)

        return model.Pij[i, j, t] == Vi_re * Iij_re + Vi_im * Iij_im

    model.cons_branch_flow_real_rule = pyo.Constraint(
        model.line, model.time, rule=branch_flow_real_rule
    )

    # @model.Constraint(model.line, model.time)
    def branch_flow_reactive_rule(model, i, j, t):
        Rij = model.line_R[i, j]
        Xij = model.line_X[i, j]
        Vi = model.bus_v[i, t]
        Vj = model.bus_v[j, t]
        V_thetai = model.bus_theta[i, t]
        V_thetaj = model.bus_theta[j, t]

        """
        # vi = Vi * pyo.cos(V_thetai) + Vi * pyo.sin(V_thetai) * 1j
        # vj = Vj * pyo.cos(V_thetaj) + Vj * pyo.sin(V_thetaj) * 1j
        # Iij = (vi - vj) / (Rij + Xij * 1j)
        # Iij_conj = pyo.conjugate(Iij)

        the above equations are equivalent to the following:
        """
        Vi_re = Vi * pyo.cos(V_thetai)
        Vi_im = Vi * pyo.sin(V_thetai)
        Vj_re = Vj * pyo.cos(V_thetaj)
        Vj_im = Vj * pyo.sin(V_thetaj)
        Vi_Vj_re = Vi_re - Vj_re
        Vi_Vj_im = Vi_im - Vj_im
        Iij_re = (Vi_Vj_re * Rij + Vi_Vj_im * Xij) / (Rij**2 + Xij**2)
        Iij_im = (Vi_Vj_im * Rij - Vi_Vj_re * Xij) / (Rij**2 + Xij**2)

        return model.Qij[i, j, t] == Vi_im * Iij_re - Vi_re * Iij_im

    model.cons_branch_flow_reactive_rule = pyo.Constraint(
        model.line, model.time, rule=branch_flow_reactive_rule
    )

    # Power balance constraints
    # @model.Constraint(model.bus, model.time)
    def bus_active_power_balance_rule(model, i, t):
        incoming = 0
        for k, l in model.line:
            if l == i:
                Rkl = model.line_R[k, l]
                Xkl = model.line_X[k, l]
                Vk = model.bus_v[l, t]
                Vl = model.bus_v[k, t]
                V_thetak = model.bus_theta[k, t]
                V_thetal = model.bus_theta[k, t]

                """
                # vi = Vi * pyo.cos(V_thetai) + Vi * pyo.sin(V_thetai) * 1j
                # vj = Vj * pyo.cos(V_thetaj) + Vj * pyo.sin(V_thetaj) * 1j
                # Iij = (vi - vj) / (Rij + Xij * 1j)
                # Iij_conj = pyo.conjugate(Iij)

                the above equations are equivalent to the following:
                """
                Vk_re = Vk * pyo.cos(V_thetak)
                Vk_im = Vk * pyo.sin(V_thetak)
                Vl_re = Vl * pyo.cos(V_thetal)
                Vl_im = Vl * pyo.sin(V_thetal)
                Vk_Vl_re = Vk_re - Vl_re
                Vk_Vl_im = Vk_im - Vl_im
                Ikl_re = (Vk_Vl_re * Rkl + Vk_Vl_im * Xkl) / (Rkl**2 + Xkl**2)
                Ikl_im = (Vk_Vl_im * Rkl - Vk_Vl_re * Xkl) / (Rkl**2 + Xkl**2)

                incoming = incoming + model.Pij[k, l, t] - (Ikl_re**2 + Ikl_im**2) * Rkl

        outgoing = sum(model.Pij[k, l, t] for (k, l) in model.line if k == i)
        return (
            model.gen_p[i, t] - model.load_p[i, t] == outgoing - incoming
        )  # incoming(after line_losses) - outgoing == gen_p - load_p

    model.cons_bus_active_power_balance_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_active_power_balance_rule
    )

    # @model.Constraint(model.bus, model.time)
    def bus_reactive_power_balance_rule(model, i, t):
        incoming = 0
        for k, l in model.line:
            if l == i:
                Rkl = model.line_R[k, l]
                Xkl = model.line_X[k, l]
                Vk = model.bus_v[l, t]
                Vl = model.bus_v[k, t]
                V_thetak = model.bus_theta[k, t]
                V_thetal = model.bus_theta[k, t]

                """
                # vi = Vi * pyo.cos(V_thetai) + Vi * pyo.sin(V_thetai) * 1j
                # vj = Vj * pyo.cos(V_thetaj) + Vj * pyo.sin(V_thetaj) * 1j
                # Iij = (vi - vj) / (Rij + Xij * 1j)
                # Iij_conj = pyo.conjugate(Iij)

                the above equations are equivalent to the following:
                """
                Vk_re = Vk * pyo.cos(V_thetak)
                Vk_im = Vk * pyo.sin(V_thetak)
                Vl_re = Vl * pyo.cos(V_thetal)
                Vl_im = Vl * pyo.sin(V_thetal)
                Vk_Vl_re = Vk_re - Vl_re
                Vk_Vl_im = Vk_im - Vl_im
                Ikl_re = (Vk_Vl_re * Rkl + Vk_Vl_im * Xkl) / (Rkl**2 + Xkl**2)
                Ikl_im = (Vk_Vl_im * Rkl - Vk_Vl_re * Xkl) / (Rkl**2 + Xkl**2)

                incoming = incoming + model.Qij[k, l, t] - (Ikl_re**2 + Ikl_im**2) * Xkl

        outgoing = sum(model.Qij[k, l, t] for (k, l) in model.line if k == i)
        return (
            model.gen_q[i, t] - model.load_q[i, t] == outgoing - incoming
        )  # incoming(after line_losses) - outgoing == gen_q - load_q

    model.cons_bus_reactive_power_balance_rule = pyo.Constraint(
        model.bus, model.time, rule=bus_reactive_power_balance_rule
    )

    # Objective
    # @model.Objective(sense=pyo.minimize)
    def obj_rule(model):
        return sum((model.bus_v[i, t] - 1) ** 2 for i in model.bus for t in model.time)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


def branch_flow_model_relax_scop(
    bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds=None
):
    slack = list(slack_bus.keys())[0]
    if bus_v_bounds is None:
        bus_v_bounds = {}
    model = pyo.ConcreteModel()

    # Sets
    model.bus = pyo.Set(initialize=bus)
    model.time = pyo.Set(initialize=time)
    model.line = pyo.Set(initialize=line_data.keys(), dimen=2)

    # Parameters
    model.load_p = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_p.get(i, 0),
    )
    model.load_q = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_q.get(i, 0),
    )

    model.line_R = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["R"]
    )
    model.line_X = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["X"]
    )

    model.gen_p_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (gen_buses.get(i, {}).get("p_min", 0), gen_buses.get(i, {}).get("p_max", 0))
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.gen_q_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (
                gen_buses.get(i, {}).get("q_min", 0),
                gen_buses.get(i, {}).get("q_max", 0),
            )
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.bus_v_bounds = pyo.Param(
        model.bus,
        within=pyo.Any,
        initialize=lambda model, i: bus_v_bounds.get(i, (0.9**2, 1.1**2)),
    )

    # Variables
    model.gen_p = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_p_bounds[i, t],
    )

    model.gen_q = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_q_bounds[i, t],
    )

    model.v_squared = pyo.Var(
        model.bus,
        model.time,
        within=pyo.NonNegativeReals,
        bounds=lambda model, i, t: model.bus_v_bounds[i],
        initialize=1.0,
    )
    model.p_flow = pyo.Var(model.line, model.time, within=pyo.Reals, initialize=0)
    model.q_flow = pyo.Var(model.line, model.time, within=pyo.Reals, initialize=0)
    model.l_squared = pyo.Var(
        model.line, model.time, within=pyo.NonNegativeReals, initialize=0
    )

    # Set slack bus voltage
    # slack = list(slack_bus.keys())[0]
    # for t in model.time:
    #     model.v_squared[slack, t].fix(slack_bus[slack]["v"] ** 2)

    # Constraints
    def power_flow_rule(model, i, j, t):
        return (
            model.p_flow[i, j, t] ** 2 + model.q_flow[i, j, t] ** 2
            <= model.v_squared[i, t] * model.l_squared[i, j, t]
        )

    model.cons_power_flow = pyo.Constraint(model.line, model.time, rule=power_flow_rule)

    def voltage_drop_rule(model, i, j, t):
        return (
            model.v_squared[j, t]
            == model.v_squared[i, t]
            - 2
            * (
                model.line_R[i, j] * model.p_flow[i, j, t]
                + model.line_X[i, j] * model.q_flow[i, j, t]
            )
            + (model.line_R[i, j] ** 2 + model.line_X[i, j] ** 2)
            * model.l_squared[i, j, t]
        )

    model.cons_voltage_drop = pyo.Constraint(
        model.line, model.time, rule=voltage_drop_rule
    )

    def active_power_balance_rule(model, i, t):
        incoming = sum(
            model.p_flow[k, l, t] - model.line_R[k, l] * model.l_squared[k, l, t]
            for k, l in model.line
            if l == i
        )
        outgoing = sum(model.p_flow[k, l, t] for k, l in model.line if k == i)
        return outgoing == model.gen_p[i, t] - model.load_p[i, t] + incoming

    model.cons_active_power_balance = pyo.Constraint(
        model.bus, model.time, rule=active_power_balance_rule
    )

    def reactive_power_balance_rule(model, i, t):
        incoming = sum(
            model.q_flow[k, l, t] - model.line_X[k, l] * model.l_squared[k, l, t]
            for k, l in model.line
            if l == i
        )
        outgoing = sum(model.q_flow[k, l, t] for k, l in model.line if k == i)
        return outgoing == model.gen_q[i, t] - model.load_q[i, t] + incoming

    model.cons_reactive_power_balance = pyo.Constraint(
        model.bus, model.time, rule=reactive_power_balance_rule
    )

    # Objective
    def obj_rule(model):
        return sum(
            (model.v_squared[i, t] - 1) ** 2 for i in model.bus for t in model.time
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


def lindistflow_model(
    bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds=None
):
    slack = list(slack_bus.keys())[0]
    if bus_v_bounds is None:
        bus_v_bounds = {}
    model = pyo.ConcreteModel()

    # Sets
    model.bus = pyo.Set(initialize=bus)
    model.time = pyo.Set(initialize=time)
    model.line = pyo.Set(initialize=line_data.keys(), dimen=2)

    # Parameters
    model.load_p = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_p.get(i, 0),
    )
    model.load_q = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Reals,
        initialize=lambda model, i, t: load_q.get(i, 0),
    )

    model.line_R = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["R"]
    )
    model.line_X = pyo.Param(
        model.line, initialize=lambda model, i, j: line_data[i, j]["X"]
    )

    model.gen_p_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (gen_buses.get(i, {}).get("p_min", 0), gen_buses.get(i, {}).get("p_max", 0))
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.gen_q_bounds = pyo.Param(
        model.bus,
        model.time,
        within=pyo.Any,
        initialize=lambda model, i, t: (
            (
                gen_buses.get(i, {}).get("q_min", 0),
                gen_buses.get(i, {}).get("q_max", 0),
            )
            if i != slack
            else (float("-inf"), float("inf"))
        ),
    )

    model.bus_v_bounds = pyo.Param(
        model.bus,
        within=pyo.Any,
        initialize=lambda model, i: bus_v_bounds.get(i, (0.9**2, 1.1**2)),
    )

    # Variables
    model.gen_p = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_p_bounds[i, t],
    )

    model.gen_q = pyo.Var(
        model.bus,
        model.time,
        within=pyo.Reals,
        bounds=lambda model, i, t: model.gen_q_bounds[i, t],
    )

    model.v_squared = pyo.Var(
        model.bus,
        model.time,
        within=pyo.NonNegativeReals,
        bounds=lambda model, i, t: model.bus_v_bounds[i],
        initialize=1.0,
    )
    model.p_flow = pyo.Var(model.line, model.time, within=pyo.Reals, initialize=0)
    model.q_flow = pyo.Var(model.line, model.time, within=pyo.Reals, initialize=0)

    # Fix slack bus voltage
    # slack = list(slack_bus.keys())[0]
    # model.v_squared[slack, 0].fix(slack_bus[slack]["v"] ** 2)

    # Power balance constraints (LinDistFlow form)
    def active_power_balance_rule(model, i, t):
        incoming = sum(model.p_flow[k, l, t] for k, l in model.line if l == i)
        outgoing = sum(model.p_flow[k, l, t] for k, l in model.line if k == i)
        return model.gen_p[i, t] - model.load_p[i, t] == outgoing - incoming

    model.cons_active_power_balance = pyo.Constraint(
        model.bus, model.time, rule=active_power_balance_rule
    )

    def reactive_power_balance_rule(model, i, t):
        incoming = sum(model.q_flow[k, l, t] for k, l in model.line if l == i)
        outgoing = sum(model.q_flow[k, l, t] for k, l in model.line if k == i)
        return model.gen_q[i, t] - model.load_q[i, t] == outgoing - incoming

    model.cons_reactive_power_balance = pyo.Constraint(
        model.bus, model.time, rule=reactive_power_balance_rule
    )

    # Voltage drop constraint (LinDistFlow form)
    def voltage_drop_rule(model, i, j, t):
        return (
            model.v_squared[j, t]
            == model.v_squared[i, t]
            - 2 * model.line_R[i, j] * model.p_flow[i, j, t]
            - 2 * model.line_X[i, j] * model.q_flow[i, j, t]
        )

    model.cons_voltage_drop = pyo.Constraint(
        model.line, model.time, rule=voltage_drop_rule
    )

    # Objective: minimize voltage deviation
    def obj_rule(model):
        return sum(
            (model.v_squared[i, t] - 1) ** 2 for i in model.bus for t in model.time
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


if __name__ == "__main__":
    # 3 bus power system parameters: https://al-roomi.org/power-flow/3-bus-systems/system-iii
    bus = ["1", "2", "3"]
    line_data = {
        ("1", "2"): {"R": 0.08, "X": 0.24},
        ("1", "3"): {"R": 0.02, "X": 0.06},
        ("2", "3"): {"R": 0.06, "X": 0.018},
    }
    # y_bus_df, got from line_data (https://home.engineering.iastate.edu/~jdm/ee458_2011/PowerFlowEquations.pdf)
    y_bus_df = pd.DataFrame(
        {
            "1": [6.25 - 18.75j, -1.25 + 3.73j, -5.0 + 15.0j],
            "2": [-1.25 + 3.73j, 2.9167 - 8.75j, -1.6667 + 5.0j],
            "3": [-5.0 + 15.0j, -1.6667 + 5.0j, 6.6667 - 20.0j],
        },
        index=["1", "2", "3"],
    )
    load_p = {"1": 0, "2": 50 / 100, "3": 60 / 100}
    load_q = {"1": 0, "2": 20 / 100, "3": 25 / 100}
    slack_bus = {"1": {"v": 1.0, "theta": 0}}
    gen_buses = {
        "2": {"p": 20 / 100, "v": 1.0, "q_min": 0, "q_max": 35 / 100},
    }
    # extra time period variable (not actually needed for a snapshot opf)
    time = range(0, 1)

    # this bus_v_bounds setting is to verify the three models are equivalent
    # so the results should be the same even though for this nonlinear problem
    bus_v_bounds = {"1": (0.9, 1.1), "2": (0.9, 1.1), "3": (0.9, 1.1)}

    bus_inj_model = bus_injection_model(
        bus, gen_buses, y_bus_df, slack_bus, load_p, load_q, time, bus_v_bounds
    )
    branch_flow_model1 = branch_flow_model_line_current(
        bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds
    )
    branch_flow_model2 = branch_flow_model_line_flow(
        bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds
    )
    bus_v_bounds = {
        "1": (0.9**2, 1.1**2),
        "2": (0.9**2, 1.1**2),
        "3": (0.9**2, 1.1**2),
    }  # now bus_v is instead squared
    branch_flow_model_socp = branch_flow_model_relax_scop(
        bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds
    )
    lindist_flow_model = lindistflow_model(
        bus, gen_buses, line_data, slack_bus, load_p, load_q, time, bus_v_bounds
    )

    # Solve the model, use the "ipopt" solver for the nonlinear problem
    bus_inj_model_solver = pyo.SolverFactory("ipopt")
    branch_flow_model1_solver = pyo.SolverFactory("ipopt")
    branch_flow_model2_solver = pyo.SolverFactory("ipopt")
    branch_flow_model_socp_solver = pyo.SolverFactory("ipopt")
    lindist_flow_model_solver = pyo.SolverFactory("ipopt")
    # bus_inj_model_solver.options["warm_start_init_point"] = "yes"
    # branch_flow_model1_solver.options["warm_start_init_point"] = "yes"
    # branch_flow_model2_solver.options["warm_start_init_point"] = "yes"
    # branch_flow_model_socp_solver.options["warm_start_init_point"] = "yes"
    # lindist_flow_model_solver.options["warm_start_init_point"] = "yes"
    tee = False
    bus_inj_model_res = bus_inj_model_solver.solve(bus_inj_model, tee=tee)
    branch_flow_model1_res = branch_flow_model1_solver.solve(
        branch_flow_model1, tee=tee
    )
    branch_flow_model2_res = branch_flow_model2_solver.solve(
        branch_flow_model2, tee=tee
    )
    branch_flow_model_socp_res = branch_flow_model_socp_solver.solve(
        branch_flow_model_socp, tee=tee
    )
    lindist_flow_model_res = lindist_flow_model_solver.solve(
        lindist_flow_model, tee=tee
    )

    # Show the voltage and angle results
    for t in bus_inj_model.time:
        print(f"Time: {t}")
        for b in bus_inj_model.bus:
            print(
                f"Bus {b}: V={bus_inj_model.bus_v[b, t].value:.4f}, theta={bus_inj_model.bus_theta[b, t].value:.4f}"
            )
        print(f"Objective: {bus_inj_model.obj():.4f}")
        print("\n")
    for t in branch_flow_model1.time:
        print(f"Time: {t}")
        for b in branch_flow_model1.bus:
            print(
                f"Bus {b}: V={branch_flow_model1.bus_v[b, t].value:.4f}, theta={branch_flow_model1.bus_theta[b, t].value:.4f}"
            )
        print(f"Objective: {branch_flow_model1.obj():.4f}")
        print("\n")
    for t in branch_flow_model2.time:
        print(f"Time: {t}")
        for b in branch_flow_model2.bus:
            print(
                f"Bus {b}: V={branch_flow_model2.bus_v[b, t].value:.4f}, theta={branch_flow_model2.bus_theta[b, t].value:.4f}"
            )
        print(f"Objective: {branch_flow_model2.obj():.4f}")
        print("\n")
    for t in branch_flow_model_socp.time:
        print(f"Time: {t}")
        for b in branch_flow_model_socp.bus:
            print(f"Bus {b}: V={branch_flow_model_socp.v_squared[b, t].value:.4f}")
        print(f"Objective: {branch_flow_model_socp.obj():.4f}")
        print("\n")
    for t in lindist_flow_model.time:
        print(f"Time: {t}")
        for b in lindist_flow_model.bus:
            print(f"Bus {b}: V={lindist_flow_model.v_squared[b, t].value:.4f}")
        print(f"Objective: {lindist_flow_model.obj():.4f}")
        print("\n")
