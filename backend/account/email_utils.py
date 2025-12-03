"""
邮件发送工具模块
"""
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags


def send_activation_email(user, activation_url):
    """
    发送激活邮件
    
    Args:
        user: User对象
        activation_url: 激活链接URL
    """
    subject = 'Ref4D - 激活您的账户'
    
    html_message = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
            <h1 style="color: white; margin: 0; font-size: 28px;">欢迎加入 Ref4D</h1>
        </div>
        <div style="background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px;">
            <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">您好 <strong>{user.username}</strong>，</p>
            <p style="font-size: 14px; color: #6b7280; line-height: 1.6;">
                感谢您注册 Ref4D 视频生成模型评测平台。请点击下方按钮激活您的账户：
            </p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{activation_url}" 
                   style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; 
                          padding: 15px 40px; 
                          text-decoration: none; 
                          border-radius: 8px; 
                          display: inline-block;
                          font-weight: bold;
                          font-size: 16px;">
                    激活账户
                </a>
            </div>
            <p style="font-size: 12px; color: #9ca3af; margin-top: 20px;">
                或复制以下链接到浏览器地址栏：<br>
                <a href="{activation_url}" style="color: #3b82f6; word-break: break-all;">{activation_url}</a>
            </p>
            <p style="font-size: 12px; color: #9ca3af; margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                此链接将在24小时后失效。如果这不是您的操作，请忽略此邮件。
            </p>
        </div>
        <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
            <p>© 2025 Ref4D. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    plain_message = f"""
    欢迎加入 Ref4D
    
    您好 {user.username}，
    
    感谢您注册 Ref4D 视频生成模型评测平台。请访问以下链接激活您的账户：
    
    {activation_url}
    
    此链接将在24小时后失效。如果这不是您的操作，请忽略此邮件。
    
    © 2025 Ref4D. All rights reserved.
    """
    
    send_mail(
        subject=subject,
        message=plain_message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        html_message=html_message,
        fail_silently=False,
    )


def send_password_reset_email(user, reset_code):
    """
    发送密码重置验证码邮件
    
    Args:
        user: User对象
        reset_code: 6位数字验证码
    """
    subject = 'Ref4D - 密码重置验证码'
    
    html_message = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
            <h1 style="color: white; margin: 0; font-size: 28px;">密码重置请求</h1>
        </div>
        <div style="background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 10px 10px;">
            <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">您好 <strong>{user.username}</strong>，</p>
            <p style="font-size: 14px; color: #6b7280; line-height: 1.6;">
                我们收到了重置您账户密码的请求。请使用以下验证码完成密码重置：
            </p>
            <div style="background: white; 
                        border: 2px dashed #e5e7eb; 
                        border-radius: 8px; 
                        padding: 20px; 
                        text-align: center; 
                        margin: 30px 0;">
                <p style="font-size: 14px; color: #6b7280; margin: 0 0 10px 0;">您的验证码是：</p>
                <p style="font-size: 36px; 
                          font-weight: bold; 
                          color: #667eea; 
                          letter-spacing: 8px; 
                          margin: 0;
                          font-family: 'Courier New', monospace;">
                    {reset_code}
                </p>
            </div>
            <p style="font-size: 12px; color: #9ca3af; margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                此验证码将在15分钟后失效。如果这不是您的操作，请立即登录您的账户修改密码。
            </p>
        </div>
        <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
            <p>© 2025 Ref4D. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    plain_message = f"""
    密码重置请求
    
    您好 {user.username}，
    
    我们收到了重置您账户密码的请求。请使用以下验证码完成密码重置：
    
    验证码：{reset_code}
    
    此验证码将在15分钟后失效。如果这不是您的操作，请立即登录您的账户修改密码。
    
    © 2025 Ref4D. All rights reserved.
    """
    
    send_mail(
        subject=subject,
        message=plain_message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        html_message=html_message,
        fail_silently=False,
    )
