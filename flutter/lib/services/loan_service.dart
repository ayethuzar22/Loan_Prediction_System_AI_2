// =============================================================
// PART 12: FLUTTER INTEGRATION
// =============================================================
// File structure:
//   lib/
//     models/loan_result.dart
//     services/loan_service.dart
//     screens/loan_form_screen.dart
//     screens/loan_result_screen.dart
//     widgets/risk_score_widget.dart

// ─────────────────────────────────────────────────────────────
// lib/models/loan_result.dart
// ─────────────────────────────────────────────────────────────
// (paste into separate file)

/*
class LoanResult {
  final bool approved;
  final double approvalProbability;
  final int riskScore;
  final String riskLevel;

  // Approved only
  final double? recommendedAmount;
  final double? interestRate;
  final double? monthlyInstallment;
  final List<String>? riskFactors;

  // Rejected only
  final List<String>? reasons;

  LoanResult({
    required this.approved,
    required this.approvalProbability,
    required this.riskScore,
    required this.riskLevel,
    this.recommendedAmount,
    this.interestRate,
    this.monthlyInstallment,
    this.riskFactors,
    this.reasons,
  });

  factory LoanResult.fromJson(Map<String, dynamic> json) {
    return LoanResult(
      approved:             json['approved'] as bool,
      approvalProbability:  (json['approval_probability'] as num).toDouble(),
      riskScore:            json['risk_score'] as int,
      riskLevel:            json['risk_level'] as String,
      recommendedAmount:    (json['recommended_amount'] as num?)?.toDouble(),
      interestRate:         (json['interest_rate'] as num?)?.toDouble(),
      monthlyInstallment:   (json['monthly_installment'] as num?)?.toDouble(),
      riskFactors:          (json['risk_factors'] as List?)?.map((e) => e.toString()).toList(),
      reasons:              (json['reasons'] as List?)?.map((e) => e.toString()).toList(),
    );
  }
}
*/

// ─────────────────────────────────────────────────────────────
// lib/services/loan_service.dart
// ─────────────────────────────────────────────────────────────

import 'package:dio/dio.dart';
import '../models/loan_result.dart';

class LoanService {
  final Dio _dio;
  static const _baseUrl = 'https://your-api.example.com/api';

  LoanService({String? baseUrl})
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl ?? _baseUrl,
          connectTimeout: const Duration(seconds: 15),
          receiveTimeout: const Duration(seconds: 30),
          headers: {'Content-Type': 'application/json'},
        )) {
    _dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
    ));
  }

  /// Set JWT token after login
  void setAuthToken(String token) {
    _dio.options.headers['Authorization'] = 'Bearer $token';
  }

  /// Submit loan application and get prediction
  Future<LoanResult> predictLoan({
    required int userId,
    required double amountRequest,
    required int month,
    required int categoryId,
  }) async {
    try {
      final response = await _dio.post(
        '/loan/predict/',
        data: {
          'user_id':        userId,
          'amount_request': amountRequest,
          'month':          month,
          'category_id':    categoryId,
        },
      );

      if (response.statusCode == 200) {
        return LoanResult.fromJson(response.data as Map<String, dynamic>);
      } else {
        throw Exception('Unexpected status: ${response.statusCode}');
      }
    } on DioException catch (e) {
      final message = e.response?.data?['error'] ?? e.message ?? 'Network error';
      throw Exception(message);
    }
  }

  /// Fetch loan history
  Future<List<Map<String, dynamic>>> getLoanHistory() async {
    final response = await _dio.get('/loan/history/');
    final loans = response.data['loans'] as List;
    return loans.cast<Map<String, dynamic>>();
  }
}
