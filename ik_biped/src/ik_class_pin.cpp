#include "ik_class_pin.hpp"

IKRobot::IKRobot()
{
}

void IKRobot::build_model(const std::string urdf_xml_string)
{
    pinocchio::urdf::buildModelFromXML(urdf_xml_string, pinocchio::JointModelFreeFlyer(), model);
    std::cout << "model nq:" << model.nq << std::endl;
    std::cout << "model nv:" << model.nv << std::endl;

    // int idx_frame;
    // idx_frame = model.getFrameId("FL_ANKLE");
    // std::cout << "model frame:" << idx_frame << std::endl;
}

int IKRobot::get_size_q()
{
    return model.nq;
}

int IKRobot::get_size_q_dot()
{
    return model.nv;
}

Eigen::VectorXd IKRobot::get_desired_q(Eigen::VectorXd q, Eigen::Vector3d pos_foot_des, double yaw_angle, std::string joint_name)
{

    pinocchio::Data data(model);

    int joint_id;
    joint_id = model.getJointId(joint_name);

    int put_to_zero_joints[5];

    if (joint_name == "FL_ANKLE")
    {
        put_to_zero_joints[0] = 11;
        put_to_zero_joints[1] = 12;
        put_to_zero_joints[2] = 13;
        put_to_zero_joints[3] = 14;
        put_to_zero_joints[4] = 15;
    }
    if (joint_name == "FR_ANKLE")
    {
        put_to_zero_joints[0] = 6;
        put_to_zero_joints[1] = 7;
        put_to_zero_joints[2] = 8;
        put_to_zero_joints[3] = 9;
        put_to_zero_joints[4] = 10;
    }

    const pinocchio::SE3 oMdes(Eigen::Matrix3d::Identity(), pos_foot_des);

    pinocchio::Data::Matrix6x J(6, model.nv);
    Eigen::Matrix<double, 5, Eigen::Dynamic> J_truncated(5, model.nv);

    J.setZero();
    J_truncated.setZero();

    Eigen::Matrix<double, 5, 1> err;
    Eigen::Matrix<double, 3, 1> err_truncated;

    Eigen::VectorXd v(model.nv);
    Eigen::Vector3d vector_x_in_foot_frame;

    double yaw_error;
    double pitch_error;

    int i;
    for (i = 0;; i++)
    {
        pinocchio::forwardKinematics(model, data, q);
        const pinocchio::SE3 dMi = oMdes.actInv(data.oMi[joint_id]);

        vector_x_in_foot_frame = data.oMi[joint_id].rotation().transpose() * Eigen::Vector3d(cos(yaw_angle), sin(yaw_angle), 0);

        pitch_error = vector_x_in_foot_frame[1];
        yaw_error = vector_x_in_foot_frame[2];

        // err = pinocchio::log6(dMi).toVector();
        // err_truncated[0] = err[0]; err_truncated[1] = err[1]; err_truncated[2] = err[2];
        err_truncated = -dMi.inverse().translation();
        err.row(0) = err_truncated.row(0);
        err.row(1) = err_truncated.row(1);
        err.row(2) = err_truncated.row(2);
        err[3] = -pitch_error;
        err[4] = -yaw_error;

        if (err.norm() < eps)
        {
            break;
        }

        if (i >= IT_MAX)
        {
            break;
        }

        pinocchio::computeJointJacobian(model, data, q, joint_id, J);
        J_truncated.block(0, 0, 3, model.nv) = J.block(0, 0, 3, model.nv);
        J_truncated.row(3) = 1 * J.row(5);
        J_truncated.row(4) = -1 * J.row(4);

        J_truncated.block(0, 0, 5, 5) = Eigen::MatrixXd::Zero(5, 5);

        J_truncated.col(put_to_zero_joints[0]).setZero();
        J_truncated.col(put_to_zero_joints[1]).setZero();
        J_truncated.col(put_to_zero_joints[2]).setZero();
        J_truncated.col(put_to_zero_joints[3]).setZero();
        J_truncated.col(put_to_zero_joints[4]).setZero();

        Eigen::Matrix<double, 5, Eigen::Dynamic> JJt;
        JJt.noalias() = J_truncated * J_truncated.transpose();
        JJt.diagonal().array() += damp;
        v.noalias() = -J_truncated.transpose() * JJt.ldlt().solve(err);

        q = pinocchio::integrate(model, q, v * DT);
    }

    return q;
}

//     Eigen::MatrixXd get_J(Eigen::VectorXd q, std::string joint_name)
//     {

//         pinocchio::Data data(model);

//         int joint_id;
//         joint_id = model.getJointId(joint_name);

//         pinocchio::Data::Matrix6x J(6, model.nv);

//         J.setZero();
//         pinocchio::computeJointJacobian(model, data, q, joint_id, J);
//         return J;
//     }

//     Eigen::MatrixXd get_Jdot_x_q_dot(Eigen::VectorXd q, Eigen::VectorXd q_dot, std::string joint_name)
//     {
//         // https://github.com/stack-of-tasks/pinocchio/issues/1395
//         pinocchio::Data data(model);
//         // pinocchio::forwardKinematics(model,data,q,q_dot,0*q_dot);
//         // int joint_id;
//         // joint_id = model.getJointId(joint_name);
//         // Eigen::MatrixBase acceleration;
//         // acceleration = data.a[joint_id];
//         // return acceleration;
//         pinocchio::Data::Matrix6x dJ(6, model.nv);
//         // dJ.fill(0.);
//         dJ.setZero();
//         int idx_joint;
//         idx_joint = model.getJointId(joint_name);

//         pinocchio::computeJointJacobiansTimeVariation(model, data, q, q_dot);
//         pinocchio::getJointJacobianTimeVariation(model, data, idx_joint, pinocchio::LOCAL, dJ);

//         return dJ;
//     }

//     Eigen::MatrixXd get_Coriolis_term(Eigen::VectorXd q, Eigen::VectorXd q_dot)
//     {
//         pinocchio::Data data(model);
//         pinocchio::computeCoriolisMatrix(model, data, q, q_dot);
//         return data.C;
//     }

//     Eigen::MatrixXd get_gravity_term(Eigen::VectorXd q)
//     {

//         pinocchio::Data data(model);

//         pinocchio::computeGeneralizedGravity(model, data, q);

//         return data.g;
//     }

//     Eigen::MatrixXd get_inertia_matrix(Eigen::VectorXd q)
//     {

//         pinocchio::Data data(model);
//         Eigen::MatrixXd mass_matrix;

//         mass_matrix = pinocchio::crba(model, data, q);

//         return mass_matrix;
//     }
// };

// int main()
// {
//     const std::string urdf_xml_string = std::string("../assets/urdf_custom_biped/custom_robot.urdf");
//     IKRobot robot_ik(urdf_xml_string);

//     int nq;
//     int nv;
//     nq = robot_ik.get_size_q();
//     nv = robot_ik.get_size_q_dot();
//     std::cout << nq << std::endl;
//     std::cout << nv << std::endl;

//     Eigen::VectorXd current_q(nq), current_q_dot(nv);
//     Eigen::Vector3d pos_foot_des;
//     pos_foot_des << 0, 0.12, -0.24;
//     current_q << 0, 0, 0.0, 0, 0, 0, 1, 0, 0, 0.2, 0, 0, 0, 0.2, 0;
//     current_q_dot << 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0.01;

//     std::string foot_name = "FL_ANKLE";
//     robot_ik.get_desired_q(current_q, pos_foot_des, foot_name);
//     robot_ik.get_J(current_q, foot_name);
//     robot_ik.get_nonlinear_effects(current_q, current_q_dot);
//     robot_ik.get_inertia_matrix(current_q);
// }

// PYBIND11_MODULE(pybind_ik, handle)
// {

//     py::class_<IKRobot>(
//         handle, "PyIKRobot")
//         .def(py::init<const std::string>())
//         .def("get_size_q", &IKRobot::get_size_q, py::return_value_policy::reference_internal)
//         .def("get_desired_q", &IKRobot::get_desired_q, py::return_value_policy::reference_internal)
//         .def("get_size_q_dot", &IKRobot::get_size_q_dot, py::return_value_policy::reference_internal)
//         .def("get_J", &IKRobot::get_J, py::return_value_policy::reference_internal)
//         .def("get_Jdot_x_q_dot", &IKRobot::get_Jdot_x_q_dot, py::return_value_policy::reference_internal)
//         .def("get_Coriolis_term", &IKRobot::get_Coriolis_term, py::return_value_policy::reference_internal)
//         .def("get_inertia_matrix", &IKRobot::get_inertia_matrix, py::return_value_policy::reference_internal)
//         .def("get_gravity_term", &IKRobot::get_gravity_term, py::return_value_policy::reference_internal);
// }
