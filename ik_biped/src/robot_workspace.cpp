#include "ik_class_pin.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#pragma GCC diagnostic pop
#include <iomanip>

const double eps = 1e-3;
const int IT_MAX = 50;
const double DT = 0.1;
const double damp = 1e-8;

IKRobot::IKRobot()
{
}

void IKRobot::build_model(const std::string urdf_xml_string)
{
    pinocchio::urdf::buildModelFromXML(urdf_xml_string, pinocchio::JointModelFreeFlyer(), model_);
    q_ = pinocchio::neutral(model_);

    std::cout << "model nq:" << model_.nq << std::endl;
    std::cout << "model nv:" << model_.nv << std::endl;
    std::cout << "model njoints:" << model_.njoints << std::endl;

    std::cout << "model frames:" << std::endl;
    for (auto f : model_.frames) {
        std::cout << "  frame name:" << f.name << std::endl;
        std::cout << "  parent joint:" << model_.names[f.parent] << std::endl;
        std::cout << "  frame placement:" << f.placement << std::endl;
        std::cout << "  frame type" << f.type << std::endl;
        std::cout << "" << std::endl;
    }
    for (auto j : model_.joints) {
        std::cout << "joint idx_q:" << j.idx_q() << std::endl;
        std::cout << "joint idx_v:" << j.idx_v() << std::endl;
        std::cout << "joint nq:" << j.nq() << std::endl;
        std::cout << "joint nv:" << j.nv() << std::endl;
        std::cout << "joint shortname:" << j.shortname() << std::endl;
        // std::cout << "joint type:" << j.type() << std::endl;
        std::cout << "" << std::endl;
    }

}

bool IKRobot::has_model() const
{
    return model_.njoints > 1;
}



void compute_robot_workspace(const std::vector<IKRobot::BodyState>& body_states, std::vector<Eigen::Vector3d> &body_positions_solution)
{
    auto base_link = std::find_if(body_states.begin(), body_states.end(), [](const BodyState& bs) {
        return bs.name == "base_link";
    });
    if (base_link == body_states.end()) {
        std::cout << "base_link not found" << std::endl;
        return std::vector<JointState>();
    }

    std::vector<pinocchio::Model::FrameIndex> body_frame_ids;
    for (const auto &body: body_states) {
        auto frame_id = model_.getFrameId(body.name);
        if (frame_id == model_.frames.size()) {
            std::cout << "body not found:" << body.name << std::endl;
            return std::vector<JointState>();
        }
        body_frame_ids.push_back(frame_id);
    }
    // check that the frames are connected to all leaf joints
    for (const auto &joint_subtree : model_.subtrees) {
        if (joint_subtree.size() == 1) { // is leaf joint
            if (std::find_if(body_frame_ids.begin(), body_frame_ids.end(),
                [&](const pinocchio::Model::FrameIndex& frame_id) {
                    return model_.frames[frame_id].parent == joint_subtree[0];
                }) == body_frame_ids.end()) {
                std::cout << "leaf joint not found:" << model_.names[joint_subtree[0]] << std::endl;
                return std::vector<JointState>();
            }
        }
    }

    // initialize q for base_link
    int base_link_joint_id = model_.frames[model_.getFrameId("base_link")].parent;
    const auto &base_link_joint = model_.joints[base_link_joint_id];
    assert(base_link_joint.nq() == 7);
    q_[base_link_joint.idx_q()] = base_link->position[0];
    q_[base_link_joint.idx_q() + 1] = base_link->position[1];
    q_[base_link_joint.idx_q() + 2] = base_link->position[2];
    q_[base_link_joint.idx_q() + 3] = base_link->orientation.x();
    q_[base_link_joint.idx_q() + 4] = base_link->orientation.y();
    q_[base_link_joint.idx_q() + 5] = base_link->orientation.z();
    q_[base_link_joint.idx_q() + 6] = base_link->orientation.w();

    Eigen::VectorXd q = q_;
    pinocchio::Data data(model_);


    std::vector<JointState> joint_states;
    for (int joint_idx = 0; joint_idx < model_.njoints; joint_idx++) {
        const auto &joint = model_.joints[joint_idx];
        if (joint.nq() == 1 && joint.idx_q() != -1) {
            JointState joint_state;
            joint_state.name = model_.names[joint_idx];
            double q_min = std::numeric_limits<double>::lowest();
            double q_max = std::numeric_limits<double>::max();
            if (model_.lowerPositionLimit.size() > joint.idx_q()
                && model_.upperPositionLimit.size() > joint.idx_q()) {
                q_min = model_.lowerPositionLimit[joint.idx_q()];
                q_max = model_.upperPositionLimit[joint.idx_q()];
            }
            q[joint.idx_q()] = fmax(q_min, fmin(q[joint.idx_q()], q_max));
            joint_state.position = q[joint.idx_q()];
            joint_states.push_back(joint_state);
        }
    }
    q_ = q;
    pinocchio::forwardKinematics(model_, data, q_);
    for (const auto &body: body_states) {
        auto frame_id = model_.getFrameId(body.name);
        const auto &cur_to_world = data.oMf[frame_id];
        body_positions_solution.push_back(cur_to_world.translation());
    }


    Eigen::MatrixXd J_for_ff_stacked(nb_constraints, model_.nv);
    J_for_ff_stacked.setZero();
    Eigen::MatrixXd body_vels_stacked(nb_constraints, 1);
    body_vels_stacked.setZero();

    int cur_constraint_ff = 0;
    for (const auto &body: body_states) {
        Eigen::Vector3d body_vel = body.linear_velocity;
        if (body_vel.hasNaN()) {
            body_vel.setZero();
        }
        auto frame_id = model_.getFrameId(body.name);
        Eigen::MatrixXd J(6, model_.nv);
        J.setZero();
        pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        if (body.type == BodyState::ContraintType::FULL_6DOF) {
            J_for_ff_stacked.block(cur_constraint_ff, 0, 6, model_.nv) = J;
            body_vels_stacked.block(cur_constraint_ff, 0, 3, 1) = body_vel;
            body_vels_stacked.block(cur_constraint_ff+3, 0, 3, 1).setZero();
            cur_constraint_ff += 6;
        } else if (body.type == BodyState::ContraintType::POS_ONLY) {
            J_for_ff_stacked.block(cur_constraint_ff, 0, 3, model_.nv) = J;
            body_vels_stacked.block(cur_constraint_ff, 0, 3, 1) = body_vel;
            cur_constraint_ff += 3;
        } else if (body.type == BodyState::ContraintType::POS_AXIS) {
            J_for_ff_stacked.block(cur_constraint_ff, 0, 5, model_.nv) = J;
            body_vels_stacked.block(cur_constraint_ff, 0, 3, 1) = body_vel;
            body_vels_stacked.block(cur_constraint_ff+3, 0, 2, 1).setZero();
            cur_constraint_ff += 5;
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> QR_ff(J_for_ff_stacked);
    Eigen::VectorXd q_vel(QR_ff.solve(body_vels_stacked));

    // std::cout << "q_vel: " << q_vel.transpose() << std::endl;
    // std::cout << "body vels: " << body_vels_stacked.transpose() << std::endl;
    for (auto &joint_state: joint_states)
    {
        auto joint_id = model_.getJointId(joint_state.name);
        auto joint = model_.joints[joint_id];
        joint_state.velocity = q_vel[joint.idx_v()];
    }

    int nb_contacts = 0;
    for (const auto &body: body_states)
    {
        if (body.name != "base_link")
        {
            if (body.in_contact == true)
            {
                nb_contacts++;
            }
        }
    }

    if (nb_contacts == 1)
    {
        Eigen::MatrixXd J_contacts;
        J_contacts.resize(5, model_.nv);
        J_contacts.setZero();

        for (const auto &body: body_states)
        {
            if (body.name != "base_link")
            {
                if (body.in_contact == true)
                {
                    auto frame_id = model_.getFrameId(body.name);
                    Eigen::MatrixXd J(6, model_.nv);
                    J.setZero();
                    pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::WORLD, J);
                    Eigen::MatrixXd J_block(5, model_.nv);
                    J_block << J.block(0, 0, 3, model_.nv), J.block(4, 0, 2, model_.nv);
                    J_contacts.block(0, 0, 5, model_.nv) = J_block;
                }
            }
        }

        Eigen::MatrixXd P(model_.nv, model_.nv);
        Eigen::MatrixXd I_nv = Eigen::MatrixXd::Identity(model_.nv, model_.nv);

        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> complete_orth_decomp(J_contacts);
        auto J_contacts_pinv = complete_orth_decomp.pseudoInverse();

        P = I_nv - J_contacts_pinv * J_contacts;
        auto PB = P * B_matrix_;
        pinocchio::computeGeneralizedGravity(model_, data, q_); // data.g

        Eigen::HouseholderQR<Eigen::MatrixXd> QR(PB);
        auto y = P * data.g;
        Eigen::VectorXd feedforward_torque(QR.solve(y));

        for (auto &joint_state: joint_states)
        {
            auto joint_id = model_.getJointId(joint_state.name);
            auto joint = model_.joints[joint_id];
            joint_state.effort = feedforward_torque[joint.idx_v() - 6];
        }
    }
    else
    {
        for (auto &joint_state: joint_states)
        {
            joint_state.effort = 0.0;

        }
    }

    // for (auto joint : joint_states)
    // {
    //     std::cout << joint.name << " " << joint.position << " " << joint.effort << std::endl;
    // }


    return joint_states;
}
