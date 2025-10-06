@echo off
echo ========================================
echo AWS EC2 Manual Setup Helper
echo ========================================
echo.

echo 🔧 This script helps you set up AWS EC2 for manual deployment
echo.

:: Check if AWS CLI is installed
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ AWS CLI is not installed.
    echo 📥 Download from: https://aws.amazon.com/cli/
    pause
    exit /b 1
)

echo ✅ AWS CLI is installed

:: Create EC2 instance
echo 🚀 Creating EC2 instance...
echo.
echo 📝 Please run these commands in your terminal:
echo.
echo 1. Create security group:
echo    aws ec2 create-security-group --group-name whatsapp-bot-sg --description "WhatsApp Bot Security Group"
echo.
echo 2. Add inbound rules:
echo    aws ec2 authorize-security-group-ingress --group-name whatsapp-bot-sg --protocol tcp --port 22 --cidr 0.0.0.0/0
echo    aws ec2 authorize-security-group-ingress --group-name whatsapp-bot-sg --protocol tcp --port 80 --cidr 0.0.0.0/0
echo    aws ec2 authorize-security-group-ingress --group-name whatsapp-bot-sg --protocol tcp --port 443 --cidr 0.0.0.0/0
echo.
echo 3. Create EC2 instance:
echo    aws ec2 run-instances ^
echo        --image-id ami-0c02fb55956c7d316 ^
echo        --count 1 ^
echo        --instance-type t3.small ^
echo        --key-name your-key-pair ^
echo        --security-groups whatsapp-bot-sg ^
echo        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=whatsapp-bot}]'
echo.
echo 4. Get public IP:
echo    aws ec2 describe-instances --filters "Name=tag:Name,Values=whatsapp-bot" --query "Reservations[].Instances[].PublicIpAddress" --output text
echo.

echo 📋 After instance creation:
echo    1. SSH into the instance
echo    2. Run the setup script: ./ec2-setup.sh
echo    3. Configure environment variables
echo    4. Start the application
echo.

pause