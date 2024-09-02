#include "ik_class_pin.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#pragma GCC diagnostic pop
#include <iomanip>

const double CLAMP_ERROR_MAG = 0.002;
const int IT_MAX = 400;
const double DT = 0.1;
const double DAMP = 1e-5;


void get_singular_values(const Eigen::MatrixXd &A, Eigen::VectorXd &S)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    S = svd.singularValues();
}

void clamp_error_per_step(Eigen::VectorXd &e, double max_magnitude)
{
    if (e.size() < 3) {
        throw std::runtime_error("e must have size > 3");
    }
    Eigen::Vector3d e_translation(e(0), e(1), e(2));
    Eigen::Vector3d e_translation_clamped;
    double magnitude = e_translation.norm();
    if (magnitude > max_magnitude) {
        e_translation_clamped = max_magnitude * e_translation / magnitude;
    } else {
        e_translation_clamped = e_translation;
    }
    for (int i = 0; i < 3; i++) {
        e(i) = e_translation_clamped(i);
    }
}

IKRobot::IKRobot()
{
}

void IKRobot::build_model(const std::string urdf_xml_string)
{
    // Create model.
    model_ = pinocchio::Model();
    pinocchio::urdf::buildModelFromXML(urdf_xml_string, pinocchio::JointModelFreeFlyer(), model_);
    q_pin_ = pinocchio::neutral(model_);
    // q_pin_ += Eigen::VectorXd::Random(model_.nq) * 0.01;

    std::cout << "model nq:" << model_.nq << std::endl;
    std::cout << "model nv:" << model_.nv << std::endl;
    std::cout << "model njoints:" << model_.njoints << std::endl;

    std::cout << "model frames:" << std::endl;
    for (auto f : model_.frames) {
        std::cout << "  frame name:" << f.name << std::endl;
        std::cout << "  parent joint:" << model_.names[f.parent] << std::endl;
        std::cout << "  frame placement:" << f.placement << std::endl;
        std::cout << "  frame type: " << f.type << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "" << std::endl;
    }

    std::cout << "model joints:" << std::endl;
    for (auto j : model_.joints) {
        std::cout << "  joint idx_q:" << j.idx_q() << std::endl;
        std::cout << "  joint idx_v:" << j.idx_v() << std::endl;
        std::cout << "  joint nq:" << j.nq() << std::endl;
        std::cout << "  joint nv:" << j.nv() << std::endl;
        std::cout << "  joint shortname:" << j.shortname() << std::endl;
        std::cout << "  joint pos limit: " << model_.lowerPositionLimit[j.idx_q()] << " " << model_.upperPositionLimit[j.idx_q()] << std::endl;
        std::cout << "  joint max velocity: " << model_.velocityLimit[j.idx_q()] << std::endl;
        std::cout << "  joint max effort: " << model_.effortLimit[j.idx_q()] << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << " " << std::endl;
    }

    // Create B matrix.
    nb_joints_actuators_ = model_.nv - 6;
    nb_u_ = nb_joints_actuators_ - 2;
    B_matrix_ = Eigen::MatrixXd::Zero(model_.nv, nb_joints_actuators_ - 2);
    B_matrix_.block(6, 0, nb_u_/2, nb_u_/2) = Eigen::MatrixXd::Identity(nb_u_/2, nb_u_/2);
    B_matrix_.block(6 + nb_joints_actuators_/2, nb_u_/2, nb_u_/2, nb_u_/2) = Eigen::MatrixXd::Identity(nb_u_/2, nb_u_/2);
}

bool IKRobot::has_model() const
{
    return model_.njoints > 1;
}

std::vector<IKRobot::JointState> IKRobot::solve(const std::vector<IKRobot::BodyState>& body_states,
                                                IKRobot::BodyState odom_baselink,
                                                std::vector<IKRobot::JointState> &encoder_joint_states,
                                                std::vector<Eigen::Vector3d> &body_positions_solution,
                                                std::vector<IKRobot::JointState> &joint_states_for_EL_eq,
                                                Eigen::VectorXd &gravity_torque,
                                                Eigen::VectorXd &coriolis_torque,
                                                Eigen::VectorXd &inertia_torque,
                                                Eigen::VectorXd &a_foot_computed)
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

    // Check that the frames are connected to all leaf joints.
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

    // Initialize q_meas.
    Eigen::VectorXd q_meas;
    q_meas = Eigen::VectorXd::Zero(model_.nq);

    // Initialize q for base_link.
    int base_link_joint_id = model_.frames[model_.getFrameId("base_link")].parent;
    const auto &base_link_joint = model_.joints[base_link_joint_id];
    assert(base_link_joint.nq() == 7);
    q_meas[base_link_joint.idx_q()] = base_link->position[0];
    q_meas[base_link_joint.idx_q() + 1] = base_link->position[1];
    q_meas[base_link_joint.idx_q() + 2] = base_link->position[2];
    q_meas[base_link_joint.idx_q() + 3] = base_link->orientation.x();
    q_meas[base_link_joint.idx_q() + 4] = base_link->orientation.y();
    q_meas[base_link_joint.idx_q() + 5] = base_link->orientation.z();
    q_meas[base_link_joint.idx_q() + 6] = base_link->orientation.w();

    q_pin_.head(7) = q_meas.head(7);

    // Solve IK.
    Eigen::VectorXd q = q_pin_;
    pinocchio::Data data(model_);

    int nb_constraints = 0;
    for (const auto &body: body_states) {
        if (body.type == BodyState::ContraintType::FULL_6DOF) {
            nb_constraints += 6;
        } else if (body.type == BodyState::ContraintType::POS_ONLY) {
            nb_constraints += 3;
        } else if (body.type == BodyState::ContraintType::POS_AXIS) {
            nb_constraints += 5;
        }
    }

    Eigen::MatrixXd J_stacked(nb_constraints, model_.nv);
    Eigen::MatrixXd err_stacked(nb_constraints, 1);

    unsigned int i = 0;
    for (i = 0; i < IT_MAX; i++) {
        pinocchio::computeJointJacobians(model_, data, q); // Also computes forward kinematics.
        pinocchio::updateFramePlacements(model_, data);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
        int cur_constraint = 0;
        for (const auto &body: body_states) {

            auto frame_id = model_.getFrameId(body.name);

            const auto &cur_to_world = data.oMf[frame_id];
            pinocchio::SE3 des_to_world = pinocchio::SE3(body.orientation, body.position);
            Eigen::MatrixXd J(6, model_.nv);
            J.setZero();
            pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL, J);

            Eigen::VectorXd err;
            err.setZero();
            Eigen::MatrixXd J_block;
            J_block.setZero();

            if (body.type == BodyState::ContraintType::FULL_6DOF) {
                pinocchio::SE3 cur_to_des = des_to_world.actInv(cur_to_world);
                err = pinocchio::log6(cur_to_des).toVector();
                clamp_error_per_step(err, CLAMP_ERROR_MAG);
                // J v = -err
                // v = - JT (J JT + DAMP I)^-1 err
                J_block = J.block(0, 0, 6, model_.nv);
                J_stacked.block(cur_constraint, 0, 6, model_.nv) = J_block;
                err_stacked.block(cur_constraint, 0, 6, 1) = err;
                cur_constraint += 6;

            } else if (body.type == BodyState::ContraintType::POS_ONLY) {
                pinocchio::SE3 des_to_cur = cur_to_world.actInv(des_to_world);
                err = - des_to_cur.translation();
                clamp_error_per_step(err, CLAMP_ERROR_MAG);
                J_block = J.block(0, 0, 3, model_.nv);
                J_stacked.block(cur_constraint, 0, 3, model_.nv) = J_block;
                err_stacked.block(cur_constraint, 0, 3, 1) = err;
                cur_constraint += 3;

            } else if (body.type == BodyState::ContraintType::POS_AXIS) {
                pinocchio::SE3 des_to_cur = cur_to_world.actInv(des_to_world);
                Eigen::Vector3d a_des_in_cur = des_to_cur.rotation() * body.align_axis;
                Eigen::Matrix<double, 3, 2> a_normal_basis;
                a_normal_basis << 0, 0, // TODO compute from a
                           1, 0,
                            0, 1;
                auto a_proj = a_normal_basis.transpose() * a_des_in_cur;
                auto a_err = -a_proj;
                Eigen::VectorXd p_err = -des_to_cur.translation();
                clamp_error_per_step(p_err, CLAMP_ERROR_MAG);

                auto J_w = J.block(3, 0, 3, model_.nv);
                auto partial_a_partial_q = J_w.colwise().cross(body.align_axis);
                auto partial_a_proj_partial_q = a_normal_basis.transpose() * partial_a_partial_q;
                J_block.resize(5, model_.nv);
                J_block << J.block(0, 0, 3, model_.nv),
                           partial_a_proj_partial_q;
                err.resize(5, 1);

                err << p_err, a_err;
                J_stacked.block(cur_constraint, 0, 5, model_.nv) = J_block;
                err_stacked.block(cur_constraint, 0, 5, 1) = err;
                cur_constraint += 5;

            }
        }
        Eigen::MatrixXd identity_mat;
        identity_mat = Eigen::MatrixXd::Identity(nb_constraints, nb_constraints);
        v = - J_stacked.transpose() * (J_stacked * J_stacked.transpose() + DAMP * identity_mat).ldlt().solve(err_stacked);
        q = pinocchio::integrate(model_, q, v * DT);
    }

    // Send the solution to the joint_trajectory.
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

    // Update q_meas with encoder_joint_state
    for (int joint_idx = 0; joint_idx < model_.njoints; joint_idx++) {
        const auto &joint = model_.joints[joint_idx];
        if (joint.nq() == 1 && joint.idx_q() != -1) {
            auto encoder_joint = std::find_if(encoder_joint_states.begin(), encoder_joint_states.end(), [&](JointState& js) {
                return js.name == model_.names[joint_idx];});
            if (encoder_joint != encoder_joint_states.end()) {
                q_meas[joint.idx_q()] = encoder_joint->position;
            }
        }
    }

    // Update q_pin_ with the actual state.
    q_pin_ = q_meas;

    pinocchio::forwardKinematics(model_, data, q_meas); // Perform fw kin. with the actual state.
    for (const auto &body: body_states) {
        auto frame_id = model_.getFrameId(body.name);
        const auto &cur_to_world = data.oMf[frame_id];
        body_positions_solution.push_back(cur_to_world.translation());
    }

    pinocchio::computeJointJacobians(model_, data, q_meas);
    Eigen::MatrixXd J_for_ff_stacked(nb_constraints, model_.nv);
    J_for_ff_stacked.setZero();
    Eigen::MatrixXd body_vels_stacked(nb_constraints, 1);
    body_vels_stacked.setZero();
    int cur_constraint_ff = 0;
    for (const auto &body: body_states)
    {
        Eigen::Vector3d body_vel = body.linear_velocity;
        if (body_vel.hasNaN()) {
            std::cout << "body_vel has NaN" << std::endl;
            body_vel.setZero();
        }

        auto frame_id = model_.getFrameId(body.name);
        Eigen::MatrixXd J(6, model_.nv);
        J.setZero();
        pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);

        if (body.type == BodyState::ContraintType::FULL_6DOF) {
            J_for_ff_stacked.block(cur_constraint_ff, 0, 6, model_.nv) = J;
            body_vels_stacked.block(cur_constraint_ff, 0, 3, 1) = body_vel;
            body_vels_stacked.block(cur_constraint_ff + 3, 0, 3, 1).setZero();
            cur_constraint_ff += 6;
        } else if (body.type == BodyState::ContraintType::POS_ONLY) {
            J_for_ff_stacked.block(cur_constraint_ff, 0, 3, model_.nv) = J.block(0, 0, 3, model_.nv);
            body_vels_stacked.block(cur_constraint_ff, 0, 3, 1) = body_vel;
            cur_constraint_ff += 3;
        } else if (body.type == BodyState::ContraintType::POS_AXIS) {
            J_for_ff_stacked.block(cur_constraint_ff, 0, 3, model_.nv) = J.block(0, 0, 3, model_.nv);
            J_for_ff_stacked.block(cur_constraint_ff + 3, 0, 2, model_.nv) = J.block(4, 0, 2, model_.nv); // TODO compute from a
            body_vels_stacked.block(cur_constraint_ff, 0, 3, 1) = body_vel;
            body_vels_stacked.block(cur_constraint_ff + 3, 0, 2, 1).setZero();
            cur_constraint_ff += 5;
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> QR_ff(J_for_ff_stacked);
    Eigen::VectorXd q_vel(QR_ff.solve(body_vels_stacked));

    // Send the solution to the joint_trajectory velocity.
    for (auto &joint_state: joint_states)
    {
        auto joint_id = model_.getJointId(joint_state.name);
        auto joint = model_.joints[joint_id];
        double q_vel_min = std::numeric_limits<double>::lowest();
        double q_vel_max = std::numeric_limits<double>::max();
        if (model_.velocityLimit.size() > joint.idx_q())
        {
            q_vel_min = -model_.velocityLimit[joint.idx_q()];
            q_vel_max = model_.velocityLimit[joint.idx_q()];
        }
        joint_state.velocity = fmax(q_vel_min, fmin(q_vel[joint.idx_v()], q_vel_max));
    }

    // Update q_vel with measured velocities from encoder and state estimation.
    Eigen::VectorXd q_vel_meas = Eigen::VectorXd::Zero(model_.nv);
    for (int joint_idx = 0; joint_idx < model_.njoints; joint_idx++) {
        const auto &joint = model_.joints[joint_idx];
        if (joint.nv() == 1 && joint.idx_v() != -1) {
            auto encoder_joint = std::find_if(encoder_joint_states.begin(), encoder_joint_states.end(), [&](JointState& js) {
                return js.name == model_.names[joint_idx];});
            if (encoder_joint != encoder_joint_states.end()) {
                q_vel_meas[joint.idx_v()] = encoder_joint->velocity;
            }
        }
    }

    q_vel_meas[base_link_joint.idx_v()] = odom_baselink.linear_velocity[0];
    q_vel_meas[base_link_joint.idx_v() + 1] = odom_baselink.linear_velocity[1];
    q_vel_meas[base_link_joint.idx_v() + 2] = odom_baselink.linear_velocity[2];
    q_vel_meas[base_link_joint.idx_v() + 3] = odom_baselink.angular_velocity[0];
    q_vel_meas[base_link_joint.idx_v() + 4] = odom_baselink.angular_velocity[1];
    q_vel_meas[base_link_joint.idx_v() + 5] = odom_baselink.angular_velocity[2];


    // Perform Inverse Dynamics.
    pinocchio::computeJointJacobiansTimeVariation(model_, data, q_meas, q_vel_meas);
    Eigen::MatrixXd body_accs_stacked(nb_constraints, 1);
    body_accs_stacked.setZero();
    cur_constraint_ff = 0;
    Eigen::MatrixXd J_dot_for_ff_stacked(nb_constraints, model_.nv);
    J_dot_for_ff_stacked.setZero();
    for (const auto &body: body_states)
    {
        auto frame_id = model_.getFrameId(body.name);
        Eigen::MatrixXd J_dot(6, model_.nv);
        J_dot.setZero();
        pinocchio::getFrameJacobianTimeVariation(model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_dot);
        Eigen::Vector3d body_acc = body.linear_acceleration;
        if (body_acc.hasNaN()){
            std::cout << "body_acc has NaN" << std::endl;
            body_acc.setZero();
        }
        // auto frame_id = model_.getFrameId(body.name);
        if (body.type == BodyState::ContraintType::FULL_6DOF) {
            J_dot_for_ff_stacked.block(cur_constraint_ff, 0, 6, model_.nv) = J_dot;
            body_accs_stacked.block(cur_constraint_ff, 0, 3, 1) = body_acc;
            body_accs_stacked.block(cur_constraint_ff + 3, 0, 3, 1).setZero();
            cur_constraint_ff += 6;
        } else if (body.type == BodyState::ContraintType::POS_ONLY) {
            J_dot_for_ff_stacked.block(cur_constraint_ff, 0, 3, model_.nv) = J_dot.block(0, 0, 3, model_.nv);
            body_accs_stacked.block(cur_constraint_ff, 0, 3, 1) = body_acc;
            cur_constraint_ff += 3;
        } else if (body.type == BodyState::ContraintType::POS_AXIS) {
            J_dot_for_ff_stacked.block(cur_constraint_ff, 0, 3, model_.nv) = J_dot.block(0, 0, 3, model_.nv);
            J_dot_for_ff_stacked.block(cur_constraint_ff + 3, 0, 2, model_.nv) = J_dot.block(4, 0, 2, model_.nv); // TODO compute from a
            body_accs_stacked.block(cur_constraint_ff, 0, 3, 1) = body_acc;
            body_accs_stacked.block(cur_constraint_ff + 3, 0, 2, 1).setZero();
            cur_constraint_ff += 5;
        }
    }
    Eigen::VectorXd q_acc(QR_ff.solve(body_accs_stacked - J_dot_for_ff_stacked * q_vel_meas));

    // create a list of bodies in contact
    std::vector<std::reference_wrapper<const IKRobot::BodyState>> bodies_in_contact;
    for (const auto &body: body_states)
    {
        if (body.in_contact == true)
        {
            bodies_in_contact.push_back(body);
        }
    }

    if (bodies_in_contact.size() == 1)
    {
        auto &body = bodies_in_contact[0].get();
        Eigen::MatrixXd J_contacts;
        Eigen::MatrixXd J_contacts_dot;

        Eigen::MatrixXd J(6, model_.nv);
        J.setZero();
        auto frame_id = model_.getFrameId(body.name);
        pinocchio::getFrameJacobian(model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        Eigen::MatrixXd J_dot(6, model_.nv);
        J_dot.setZero();
        pinocchio::getFrameJacobianTimeVariation(model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_dot);

        if (bodies_in_contact[0].get().type == BodyState::ContraintType::FULL_6DOF)
        {
            J_contacts.resize(6, model_.nv);
            J_contacts.setZero();
            J_contacts.block(0, 0, 6, model_.nv) = J;
            J_contacts_dot.resize(6, model_.nv);
            J_contacts_dot.setZero();
            J_contacts_dot.block(0, 0, 6, model_.nv) = J_dot;
        }
        else if (bodies_in_contact[0].get().type == BodyState::ContraintType::POS_ONLY)
        {
            std::cout << "Not implemented" << std::endl;
            assert(false); // TODO not implemented
        }
        else if (bodies_in_contact[0].get().type == BodyState::ContraintType::POS_AXIS)
        {
            J_contacts.resize(5, model_.nv);
            J_contacts.setZero();

            Eigen::Matrix<double, 3, 2> a_normal_basis;
            a_normal_basis << 0, 0, // TODO compute from a
                        1, 0,
                        0, 1;
            auto J_w = J.block(3, 0, 3, model_.nv);
            auto partial_a_partial_q = J_w.colwise().cross(body.align_axis);
            auto partial_a_proj_partial_q = a_normal_basis.transpose() * partial_a_partial_q;
            J_contacts.block(0, 0, 5, model_.nv) << J.block(0, 0, 3, model_.nv),
                                                    partial_a_proj_partial_q;
            J_contacts_dot.resize(5, model_.nv);
            J_contacts_dot.setZero();
            J_contacts_dot.block(0, 0, 3, model_.nv) = J_dot.block(0, 0, 3, model_.nv);
            J_contacts_dot.block(0 + 3, 0, 2, model_.nv) = J_dot.block(4, 0, 2, model_.nv); // TODO compute from a
        }
        Eigen::MatrixXd P(model_.nv, model_.nv);
        Eigen::MatrixXd I_nv = Eigen::MatrixXd::Identity(model_.nv, model_.nv);

        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> complete_orth_decomp(J_contacts);
        auto J_contacts_pinv = complete_orth_decomp.pseudoInverse();
        P = I_nv - J_contacts_pinv * J_contacts;
        pinocchio::computeGeneralizedGravity(model_, data, q_meas); // data.g
        pinocchio::computeCoriolisMatrix(model_, data, q_meas, q_vel_meas); // data.C
        pinocchio::crba(model_, data, q_meas); // data.M
        data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
        auto y = P * data.g + P * data.C * q_vel_meas + P * data.M * q_acc;

        Eigen::MatrixXd PB = P * B_matrix_;
        Eigen::HouseholderQR<Eigen::MatrixXd> QR(PB);
        Eigen::VectorXd feedforward_torque(QR.solve(y));
        Eigen::VectorXd feedforward_torque_all_joints = Eigen::VectorXd::Zero(nb_joints_actuators_);
        feedforward_torque_all_joints.head(nb_u_/2) = feedforward_torque.head(nb_u_/2);
        feedforward_torque_all_joints.segment((nb_u_)/2 + 1, (nb_u_)/2) = feedforward_torque.tail(nb_u_/2);

        // for debugging:
        auto y_with_acceleration = P * data.g + P * data.C * q_vel_meas + P * data.M * q_acc;
        Eigen::VectorXd feedforward_torque_with_acceleration(QR.solve(y_with_acceleration));
        gravity_torque = QR.solve(P * data.g);
        coriolis_torque = QR.solve(P * data.C * q_vel_meas);
        inertia_torque = QR.solve(P * data.M * q_acc);

        // compute contact forces:
        Eigen::MatrixXd block_matrix_q_acc_and_lambda = Eigen::MatrixXd::Zero(model_.nv + 5, model_.nv + 5);
        block_matrix_q_acc_and_lambda.block(0, 0, model_.nv, model_.nv) = data.M;
        block_matrix_q_acc_and_lambda.block(model_.nv, 0, 5, model_.nv) = J_contacts;
        block_matrix_q_acc_and_lambda.block(0, model_.nv, model_.nv, 5) = -J_contacts.transpose();

        Eigen::VectorXd right_hand_side_q_lambda = Eigen::MatrixXd::Zero(model_.nv + 5, 1);
        right_hand_side_q_lambda.head(model_.nv) = - data.g - data.C * q_vel_meas + B_matrix_ * feedforward_torque_with_acceleration;
        right_hand_side_q_lambda.tail(5) = -J_contacts_dot * q_vel;

        Eigen::HouseholderQR<Eigen::MatrixXd> QR_q_lambda(block_matrix_q_acc_and_lambda);
        Eigen::VectorXd q_acc_and_lambda(QR_q_lambda.solve(right_hand_side_q_lambda));

        auto computed_q_acc = q_acc_and_lambda.head(model_.nv);
        // compute a_foot:
        a_foot_computed = J_contacts * computed_q_acc + J_contacts_dot * q_vel_meas;

        for (auto &joint_state: joint_states)
        {
            auto joint_id = model_.getJointId(joint_state.name);
            auto joint = model_.joints[joint_id];
            double q_effort_min = std::numeric_limits<double>::lowest();
            double q_effort_max = std::numeric_limits<double>::max();
            if (model_.effortLimit.size() > joint.idx_q())
            {
                q_effort_min = -model_.effortLimit[joint.idx_q()];
                q_effort_max = model_.effortLimit[joint.idx_q()];
            }
            joint_state.effort = fmax(q_effort_min, fmin(feedforward_torque_all_joints[joint.idx_v() - 6], q_effort_max));

        }
        for (int joint_idx = 0; joint_idx < model_.njoints; joint_idx++)
        {
            const auto &joint = model_.joints[joint_idx];
            if (joint.nq() == 1 && joint.idx_q() != -1) {
                JointState joint_state_for_EL_eq;
                joint_state_for_EL_eq.name = model_.names[joint_idx];
                joint_state_for_EL_eq.position = q_meas[joint.idx_q()];
                joint_state_for_EL_eq.velocity = q_vel_meas[joint.idx_v()];
                joint_state_for_EL_eq.acceleration = q_acc[joint.idx_v()];
                joint_states_for_EL_eq.push_back(joint_state_for_EL_eq);
            }
        }
    }
    else
    {
        for (auto &joint_state: joint_states)
        {
            joint_state.effort = 0.0;
        }
    }

    return joint_states;
}
