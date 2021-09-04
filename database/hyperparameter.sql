-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Sep 04, 2021 at 04:49 PM
-- Server version: 10.4.13-MariaDB
-- PHP Version: 7.4.8

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `hyperparameter`
--

-- --------------------------------------------------------

--
-- Table structure for table `parameter`
--

CREATE TABLE `parameter` (
  `id_parameter` int(11) NOT NULL,
  `p` int(11) NOT NULL,
  `d` int(11) NOT NULL,
  `q` int(11) NOT NULL,
  `P_seasonal` int(11) NOT NULL,
  `D_seasonal` int(11) NOT NULL,
  `Q_seasonal` int(11) NOT NULL,
  `S_seasonal` int(11) NOT NULL,
  `kategori` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `parameter`
--

INSERT INTO `parameter` (`id_parameter`, `p`, `d`, `q`, `P_seasonal`, `D_seasonal`, `Q_seasonal`, `S_seasonal`, `kategori`) VALUES
(1, 1, 1, 1, 0, 0, 0, 0, 'Imigran Ilegal'),
(2, 1, 1, 2, 0, 0, 0, 0, 'Pelanggaran Batas'),
(3, 1, 1, 2, 0, 0, 0, 0, 'Pelanggaran Peraturan Pelayaran'),
(4, 2, 1, 2, 0, 0, 0, 0, 'Pembalakan Liar'),
(5, 1, 1, 1, 0, 0, 0, 0, 'Penambangan Ilegal'),
(6, 1, 1, 2, 0, 0, 0, 0, 'Pencurian BMKT'),
(7, 1, 1, 1, 0, 0, 0, 0, 'Penyelundupan'),
(8, 1, 1, 1, 0, 0, 0, 0, 'Perdagangan Manusia'),
(9, 1, 1, 1, 0, 0, 0, 0, 'Perompakan'),
(10, 1, 1, 1, 0, 0, 0, 0, 'Pertikaian di Laut'),
(11, 1, 1, 2, 0, 0, 0, 0, 'Perusakan Ekosistem'),
(12, 2, 1, 2, 0, 0, 0, 0, 'Piracy'),
(13, 1, 1, 1, 0, 0, 0, 0, 'Sabotase'),
(14, 1, 1, 1, 0, 0, 0, 0, 'Survei Hidros Ilegal'),
(15, 1, 1, 1, 0, 0, 0, 0, 'Tindak Terror di Laut'),
(16, 1, 1, 2, 0, 0, 0, 0, 'TKI Ilegal'),
(17, 1, 1, 2, 0, 0, 0, 0, 'UII Fishing');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `parameter`
--
ALTER TABLE `parameter`
  ADD PRIMARY KEY (`id_parameter`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `parameter`
--
ALTER TABLE `parameter`
  MODIFY `id_parameter` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
