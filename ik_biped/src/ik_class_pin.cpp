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
const int IT_MAX = 500;
const double DT = 0.1;
const double damp = 1e-8;

IKRobot::IKRobot()
{
}

void IKRobot::build_model(const std::string urdf_xml_string)
{
    // TODO clear previous model
    pinocchio::urdf::buildModelFromXML(urdf_xml_string, pinocchio::JointModelFreeFlyer(), model_);
    q_ = pinocchio::neutral(model_);
    // q_ += Eigen::VectorXd::Random(model_.nq) * 0.01;

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



std::vector<IKRobot::JointState> IKRobot::solve(const std::vector<IKRobot::BodyState>& body_states, std::vector<Eigen::Vector3d> &body_positions_solution)
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

    // solve IK
    Eigen::VectorXd q = q_;
    // std::cout << "initial q:" << q.transpose() << std::endl;
    pinocchio::Data data(model_);
    unsigned int i = 0;
    for (i = 0; i < IT_MAX; i++) {
        pinocchio::computeJointJacobians(model_, data, q); // also computes forward kinematics
        pinocchio::updateFramePlacements(model_, data);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
        for (const auto &body: body_states) {
            // std::cout << "body:" << body.name << std::endl;
            auto frame_id = model_.getFrameId(body.name);
            const auto &cur_to_world = data.oMf[frame_id];
            pinocchio::SE3 des_to_world = pinocchio::SE3(body.orientation, body.position);
            Eigen::MatrixXd J(6, model_.nv);
            J.setZero();
            pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL, J);
            // std::cout << "  J:" << J << std::endl;
            if (body.type == BodyState::ContraintType::FULL_6DOF) {
                pinocchio::SE3 cur_to_des = des_to_world.actInv(cur_to_world);
                auto err = pinocchio::log6(cur_to_des).toVector();
                // J v = -err
                // v = - JT (J JT + damp I)^-1 err
                Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
                v -= J.transpose() * (J * J.transpose() + damp * I6).ldlt().solve(err);
                // std::cout << " 6dof err:" << err.transpose() << std::endl;
            } else if (body.type == BodyState::ContraintType::POS_ONLY) {
                pinocchio::SE3 des_to_cur = cur_to_world.actInv(des_to_world);
                auto p_err = - des_to_cur.translation();
                auto J_block = J.block(0, 0, 3, model_.nv);
                Eigen::Matrix<double, 3, 3> I3 = Eigen::Matrix<double, 3, 3>::Identity();
                v -= J_block.transpose() * (J_block * J_block.transpose() + damp * I3).ldlt().solve(p_err);
                // std::cout << " pos err:" << p_err.transpose() << std::endl;
            } else if (body.type == BodyState::ContraintType::POS_AXIS) {
                pinocchio::SE3 des_to_cur = cur_to_world.actInv(des_to_world);
                Eigen::Vector3d a_des_in_cur = des_to_cur.rotation() * body.align_axis;
                Eigen::Matrix<double, 3, 2> a_normal_basis;
                a_normal_basis << 0, 0, // TODO compute from a
                           1, 0,
                            0, 1;
                auto a_proj = a_normal_basis.transpose() * a_des_in_cur;
                auto a_err = -a_proj;
                auto p_err = -des_to_cur.translation();
                auto J_w = J.block(3, 0, 3, model_.nv);
                auto partial_a_partial_q = J_w.colwise().cross(body.align_axis);
                auto partial_a_proj_partial_q = a_normal_basis.transpose() * partial_a_partial_q;
                Eigen::MatrixXd J_block(5, model_.nv);
                J_block << J.block(0, 0, 3, model_.nv),
                           partial_a_proj_partial_q;
                Eigen::Matrix<double, 5, 1> err;
                err << p_err, a_err;
                Eigen::Matrix<double, 5, 5> I5 = Eigen::Matrix<double, 5, 5>::Identity();
                v -= J_block.transpose() * (J_block * J_block.transpose() + damp * I5).ldlt().solve(err);
                // std::cout << " pos axis err:" << err.transpose() << std::endl;
            }
        }
        // std::cout << "v:" << v.transpose() << std::endl;
        // std::cout << "q:" << q.transpose() << std::endl;
        // std::cout << v.norm() << std::endl;
        q = pinocchio::integrate(model_, q, v * DT);
    }
    // std::cout << "iterations : " << i << "  out of " << IT_MAX << std::endl;

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

    // std::cout << std::fixed;
    // std::cout << std::setprecision(3);

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

    // std::cout << "nb_contacts:" << nb_contacts << std::endl;
    if (nb_contacts == 1)
    {
        Eigen::MatrixXd J_contacts;
        J_contacts.resize(5*nb_contacts, model_.nv);
        J_contacts.setZero();

        int i = 0;
        for (const auto &body: body_states)
        {
            if (body.name != "base_link")
            {
                if (body.in_contact == true)
                {
                    auto frame_id = model_.getFrameId(body.name);
                    Eigen::MatrixXd J(6, model_.nv);
                    J.setZero();
                    pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL, J);
                    Eigen::MatrixXd J_block(5, model_.nv);
                    J_block << J.block(0, 0, 3, model_.nv), J.block(4, 0, 2, model_.nv);
                    J_contacts.block(5 * i, 0, 5, model_.nv) = J_block;
                    i++;
                }
            }
        }

        Eigen::MatrixXd P(model_.nv, model_.nv);
        Eigen::MatrixXd I_nv = Eigen::MatrixXd::Identity(model_.nv, model_.nv);
        P = I_nv - J_contacts.transpose() * (J_contacts * J_contacts.transpose()).inverse() * J_contacts;
        // std::cout << "P:" << P << std::endl;


        auto joints_actuators = model_.njoints - 2;
        Eigen::MatrixXd B_matrix = Eigen::MatrixXd::Zero(model_.nv, joints_actuators);
        B_matrix.block(6, 0, joints_actuators, joints_actuators) = Eigen::MatrixXd::Identity(joints_actuators, joints_actuators);
        // std::cout << "B_matrix:" << B_matrix << std::endl;

        auto PB = P * B_matrix;
        Eigen::EigenSolver<Eigen::MatrixXd> eigensolver_P;
        eigensolver_P.compute(P);
        Eigen::VectorXd eigen_values = eigensolver_P.eigenvalues().real();

        pinocchio::computeGeneralizedGravity(model_, data, q_); // data.g

        auto feedforward_torque = (PB.transpose() * PB).inverse() * PB.transpose() * P * data.g;

        // todo write the below code more robustly
        for (auto &joint_state: joint_states)
        {
            auto joint_id = model_.getJointId(joint_state.name);
            if (joint_id != 0 && joint_id != 1)
            {
                joint_state.effort = feedforward_torque[joint_id - 2];
            }
        }
    }

    if (nb_contacts == 0)
    {
        for (auto &joint_state: joint_states)
        {
            joint_state.effort = std::numeric_limits<double>::quiet_NaN(); // todo check
        }
    }

    // for (auto joint : joint_states)
    // {
    //     std::cout << joint.name << " " << joint.position << " " << joint.effort << std::endl;
    // }


    return joint_states;
}
