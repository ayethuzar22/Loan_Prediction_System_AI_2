// =============================================================
// PART 12: FLUTTER SCREENS
// =============================================================

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:math' as math;
import '../services/loan_service.dart';
import '../models/loan_result.dart';

// ─────────────────────────────────────────────────────────────
// LOAN FORM SCREEN
// ─────────────────────────────────────────────────────────────

class LoanFormScreen extends StatefulWidget {
  final int userId;
  final LoanService loanService;

  const LoanFormScreen({
    Key? key,
    required this.userId,
    required this.loanService,
  }) : super(key: key);

  @override
  State<LoanFormScreen> createState() => _LoanFormScreenState();
}

class _LoanFormScreenState extends State<LoanFormScreen> {
  final _formKey     = GlobalKey<FormState>();
  final _amountCtrl  = TextEditingController();
  final _monthCtrl   = TextEditingController();
  int  _categoryId   = 1;
  bool _isLoading    = false;

  static const _categories = [
    {'id': 1, 'name': 'Personal Loan'},
    {'id': 2, 'name': 'Business Loan'},
    {'id': 3, 'name': 'Agriculture Loan'},
    {'id': 4, 'name': 'Emergency Loan'},
  ];

  Future<void> _submitApplication() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final result = await widget.loanService.predictLoan(
        userId:        widget.userId,
        amountRequest: double.parse(_amountCtrl.text.replaceAll(',', '')),
        month:         int.parse(_monthCtrl.text),
        categoryId:    _categoryId,
      );

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => LoanResultScreen(result: result),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: Colors.red.shade700,
        ),
      );
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: AppBar(
        title: const Text('Loan Application'),
        backgroundColor: const Color(0xFF1A237E),
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionHeader('Loan Details'),
              const SizedBox(height: 16),

              // Loan Amount
              _buildTextField(
                controller: _amountCtrl,
                label: 'Requested Amount (MMK)',
                hint: 'e.g. 2000000',
                keyboardType: TextInputType.number,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                validator: (v) {
                  if (v == null || v.isEmpty) return 'Enter amount';
                  final val = double.tryParse(v);
                  if (val == null || val <= 0) return 'Enter valid amount';
                  if (val < 100000) return 'Minimum 100,000 MMK';
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Tenure
              _buildTextField(
                controller: _monthCtrl,
                label: 'Loan Duration (Months)',
                hint: 'e.g. 24',
                keyboardType: TextInputType.number,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                validator: (v) {
                  if (v == null || v.isEmpty) return 'Enter duration';
                  final val = int.tryParse(v);
                  if (val == null || val < 1 || val > 60) return '1–60 months';
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Category
              _sectionLabel('Loan Category'),
              const SizedBox(height: 8),
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<int>(
                    value: _categoryId,
                    isExpanded: true,
                    items: _categories.map((cat) {
                      return DropdownMenuItem<int>(
                        value: cat['id'] as int,
                        child: Text(cat['name'] as String),
                      );
                    }).toList(),
                    onChanged: (v) => setState(() => _categoryId = v!),
                  ),
                ),
              ),

              const SizedBox(height: 32),

              // Submit button
              SizedBox(
                width: double.infinity,
                height: 54,
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _submitApplication,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A237E),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          width: 24, height: 24,
                          child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2,
                          ),
                        )
                      : const Text(
                          'Submit Application',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionHeader(String title) => Text(
        title,
        style: const TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          color: Color(0xFF1A237E),
        ),
      );

  Widget _sectionLabel(String label) => Text(
        label,
        style: const TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: Color(0xFF424242),
        ),
      );

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    TextInputType? keyboardType,
    List<TextInputFormatter>? inputFormatters,
    String? Function(String?)? validator,
  }) =>
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel(label),
          const SizedBox(height: 8),
          TextFormField(
            controller: controller,
            keyboardType: keyboardType,
            inputFormatters: inputFormatters,
            validator: validator,
            decoration: InputDecoration(
              hintText: hint,
              filled: true,
              fillColor: Colors.white,
              contentPadding: const EdgeInsets.all(16),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.grey.shade300),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.grey.shade300),
              ),
            ),
          ),
        ],
      );

  @override
  void dispose() {
    _amountCtrl.dispose();
    _monthCtrl.dispose();
    super.dispose();
  }
}


// ─────────────────────────────────────────────────────────────
// LOAN RESULT SCREEN
// ─────────────────────────────────────────────────────────────

class LoanResultScreen extends StatelessWidget {
  final LoanResult result;
  const LoanResultScreen({Key? key, required this.result}) : super(key: key);

  Color get _riskColor {
    switch (result.riskLevel) {
      case 'LOW':    return const Color(0xFF2E7D32);
      case 'MEDIUM': return const Color(0xFFF57F17);
      case 'HIGH':   return const Color(0xFFC62828);
      default:       return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F5F5),
      appBar: AppBar(
        title: const Text('Application Result'),
        backgroundColor: result.approved
            ? const Color(0xFF1B5E20)
            : const Color(0xFFB71C1C),
        foregroundColor: Colors.white,
        automaticallyImplyLeading: false,
        actions: [
          IconButton(
            icon: const Icon(Icons.home),
            onPressed: () =>
                Navigator.popUntil(context, (route) => route.isFirst),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // Status banner
            _StatusBanner(approved: result.approved),
            const SizedBox(height: 20),

            // Risk Score gauge
            RiskScoreWidget(
              score:     result.riskScore,
              riskLevel: result.riskLevel,
              color:     _riskColor,
            ),
            const SizedBox(height: 20),

            // Probability bar
            _ProbabilityBar(probability: result.approvalProbability),
            const SizedBox(height: 20),

            // Approved details
            if (result.approved) ...[
              _ApprovedDetails(result: result),
              if (result.riskFactors != null && result.riskFactors!.isNotEmpty)
                _FactorsList(
                  title: 'Risk Factors to Monitor',
                  items: result.riskFactors!,
                  icon: Icons.warning_amber,
                  color: _riskColor,
                ),
            ],

            // Rejected reasons
            if (!result.approved && result.reasons != null)
              _FactorsList(
                title: 'Reasons for Rejection',
                items: result.reasons!,
                icon: Icons.cancel_outlined,
                color: const Color(0xFFC62828),
              ),

            const SizedBox(height: 32),
            _ActionButtons(approved: result.approved),
          ],
        ),
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  final bool approved;
  const _StatusBanner({required this.approved});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: approved
            ? const Color(0xFF1B5E20)
            : const Color(0xFFB71C1C),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Icon(
            approved ? Icons.check_circle : Icons.cancel,
            color: Colors.white,
            size: 56,
          ),
          const SizedBox(height: 12),
          Text(
            approved ? 'Loan Approved!' : 'Loan Rejected',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            approved
                ? 'Congratulations! Your loan application has been approved.'
                : 'Your application did not meet our criteria at this time.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 14),
          ),
        ],
      ),
    );
  }
}

class _ProbabilityBar extends StatelessWidget {
  final double probability;
  const _ProbabilityBar({required this.probability});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Approval Probability',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              Text('${(probability * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: probability,
              minHeight: 12,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation<Color>(
                probability >= 0.7
                    ? const Color(0xFF2E7D32)
                    : probability >= 0.5
                        ? const Color(0xFFF57F17)
                        : const Color(0xFFC62828),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ApprovedDetails extends StatelessWidget {
  final LoanResult result;
  const _ApprovedDetails({required this.result});

  String _formatMMK(double amount) =>
      '${(amount / 1000).toStringAsFixed(0)}K MMK';

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.only(bottom: 20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Loan Details',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 16),
          _DetailRow('Recommended Amount',
              _formatMMK(result.recommendedAmount ?? 0)),
          _DetailRow('Monthly Interest Rate',
              '${result.interestRate?.toStringAsFixed(1) ?? '-'}%'),
          _DetailRow('Monthly Installment (EMI)',
              _formatMMK(result.monthlyInstallment ?? 0)),
        ],
      ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;
  const _DetailRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey.shade600)),
          Text(value,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
        ],
      ),
    );
  }
}

class _FactorsList extends StatelessWidget {
  final String title;
  final List<String> items;
  final IconData icon;
  final Color color;
  const _FactorsList({
    required this.title,
    required this.items,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.only(bottom: 20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 8),
            Text(title,
                style: TextStyle(
                    fontWeight: FontWeight.bold, color: color, fontSize: 15)),
          ]),
          const SizedBox(height: 12),
          ...items.map((item) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.circle, color: color, size: 8),
                    const SizedBox(width: 10),
                    Expanded(
                        child: Text(item,
                            style:
                                const TextStyle(fontSize: 14, height: 1.4))),
                  ],
                ),
              )),
        ],
      ),
    );
  }
}

class _ActionButtons extends StatelessWidget {
  final bool approved;
  const _ActionButtons({required this.approved});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (!approved)
          SizedBox(
            width: double.infinity,
            height: 50,
            child: ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.edit),
              label: const Text('Revise Application'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF1A237E),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          height: 50,
          child: OutlinedButton.icon(
            onPressed: () =>
                Navigator.popUntil(context, (route) => route.isFirst),
            icon: const Icon(Icons.home),
            label: const Text('Back to Home'),
            style: OutlinedButton.styleFrom(
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ),
      ],
    );
  }
}


// ─────────────────────────────────────────────────────────────
// RISK SCORE WIDGET (circular gauge)
// ─────────────────────────────────────────────────────────────

class RiskScoreWidget extends StatelessWidget {
  final int riskScore;
  final String riskLevel;
  final Color color;

  const RiskScoreWidget({
    Key? key,
    required this.score,
    required this.riskLevel,
    required this.color,
  })  : riskScore = score,
        super(key: key);

  final int score;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          const Text('Risk Score',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 16),
          SizedBox(
            height: 140,
            width: 140,
            child: CustomPaint(
              painter: _RiskGaugePainter(
                score: riskScore,
                color: color,
              ),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      '$riskScore',
                      style: TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                        color: color,
                      ),
                    ),
                    Text(
                      riskLevel,
                      style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: color),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _LegendDot(color: const Color(0xFF2E7D32), label: '0–30 LOW'),
              const SizedBox(width: 12),
              _LegendDot(color: const Color(0xFFF57F17), label: '31–60 MEDIUM'),
              const SizedBox(width: 12),
              _LegendDot(color: const Color(0xFFC62828), label: '61–100 HIGH'),
            ],
          ),
        ],
      ),
    );
  }
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Row(children: [
        Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 10)),
      ]);
}

class _RiskGaugePainter extends CustomPainter {
  final int score;
  final Color color;
  _RiskGaugePainter({required this.score, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = math.min(size.width, size.height) / 2 - 8;

    final bgPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 12
      ..color = Colors.grey.shade200;

    final fgPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 12
      ..color = color
      ..strokeCap = StrokeCap.round;

    const startAngle = -math.pi * 0.75;
    const sweepTotal = math.pi * 1.5;
    final sweep = sweepTotal * (score / 100);

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle, sweepTotal, false, bgPaint,
    );
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle, sweep, false, fgPaint,
    );
  }

  @override
  bool shouldRepaint(_RiskGaugePainter old) =>
      old.score != score || old.color != color;
}
